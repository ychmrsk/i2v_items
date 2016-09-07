#! /usr/bin/env python
# -*- coding:utf-8 -*-

import logging
import sys
import os
import heapq
from timeit import default_timer
from copy import deepcopy
from collections import defaultdict
import threading
import itertools

from queue import Queue, Empty

import numpy
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray

# from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from smart_open import smart_open
from types import GeneratorType

from six import string_types

from gensim.utils import SaveLoad


from gensim.matutils import unitvec, argsort

logger = logging.getLogger(__name__)


try:
    from gensim.models.word2vec_inner import FAST_VERSION, MAX_WORDS_IN_BATCH
    from gensim.models.word2vec_inner import train_batch_sg, train_batch_cbow
    logger.debug('Fast version of {0} is being used'.format(__name__))
except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    logger.warning('Slow version of {0} is being used'.format(__name__))
    FAST_VERSION = -1
    MAX_WORDS_IN_BATCH = 10000

    def train_batch_sg(model, sentences, alpha, work=None):
        """
        """
        result = 0
        for sentence in sentences:
            word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and
                           model.vocab[w].sample_int > model.random.rand() * 2**32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)
                # now go over all words from the (reduced) window, predicting each one in turn
                start = max(0, pos - model.window + reduced_window)
                for pos2, word2 in enumerate(word_vocabs[start :
                                                         (pos + model.window + 1 - reduced_window)],
                                             start):
                    # don't train on the `word` itself
                    if pos2 != pos:
                        train_sg_pair(model, model.index2word[word.index], word2.index, alpha)
            result += len(word_vocabs)
        return result

    def train_batch_cbow(model, sentences, alpha, work=None, neu1=None):
        """
        """
        result = 0
        for sentence in sentences:
            word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and
                           model.vocab[w].sample_int > model.random.rand() * 2**32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)
                start = max(0, pos - model.window + reduced_window)
                window_pos = enumerate(word_vocabs[start :
                                                   (pos + model.window + 1 - reduced_window)],
                                       start)
                word2_indices = [word2.index for pos2, word2 in window_pos
                                 if (word2 is not None and pos2 != pos)]
                l1 = np_sum(model.syn0[word2_indices], axis=0)
                if word2_indices and model.cbow_mean:
                    l1 /= len(word2_indices)
                train_cbow_pair(model, word, word2_indices, l1, alpha)
            result += len(word_vocabs)
        return result

def train_sg_pair(model, word, context_index, alpha, learn_vectors=True, learn_hidden=True,
                  context_vectors=None, context_locks=None):
    if context_vectors is None:
        context_vectors = model.syn0
    if context_locks is None:
        context_locks = model.syn0_lockf

    if word not in model.vocab:
        return
    predict_word = model.vocab[word]  # target word (NN input)

    l1 = context_vectors[context_index]  # input word (NN input/projection layer)
    lock_factor = context_locks[context_index]  # default: ones(); zeors suppress learning

    neu1e = zeros(l1.shape)

    if model.hs:
        l2a = deepcopy(model.syn1[predict_word.point])
        fa = 1.0 / (1.0 + exp(-dot(l1, l2a.T)))  # predict probability of that word is code [0] at a node 
        ga = (1 - predict_word.code - fa) * alpha  # (grand_truth - prediction) * learning_rate
        if learn_hidden:
            model.syn1[predict_word.point] += outer(ga, l1)  # update W'(hidden->output of NN)
        neu1e += dot(ga, l2a)  # save error

    if model.negative:
        word_indices = [predict_word.index]  # positive sample
        while len(word_indices) < model.negative + 1:
            w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
            if w != predict_word.index:
                word_indices.append(w)  # negative sample
        l2b = model.syn1neg[word_indices]
        fb = 1. / (1. + exp(-dot(l1, l2b.T)))
        gb = (model.neg_labels - fb) * alpha  # neg_labels = [1, 0, ... , 0]
        if learn_hidden:
            model.syn1neg[word_indices] += outer(gb, l1)
        neu1e += dot(gb, l2b)

    if learn_vectors:
        # update W (input -> hidden of NN)
        l1 += neu1e * lock_factor
    return neu1e

def train_cbow_pair(model, word, input_word_indices, l1, alpha, learn_vectors=True, learn_hidden=True):
    neu1e = zeros(l1.shape)

    if model.hs:
        l2a = model.syn1[word.point]
        fa = 1. / (1. + exp(-dot(l1, l2a.T)))
        ga = (1. - word.code -fa) * alpha  # (ground_truth - prediction) * learning_rate
        if learn_hidden:
            model.syn1[word.point] += outer(ga, l1)  # update W' (hidden -> output of NN)
        neu1e += dot(ga, l2a)  # save error

    if model.negative:
        word_indices = [word.index]
        while len(word_indices) < model.negative + 1:
            w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
            if w != word.index:
                word_indices.append(w)
        l2b = model.syn1neg[word_indices]
        fb = 1. / (1. + exp(-dot(l1, l2b.T)))
        gb = (model.neg_labels - fb) * alpha
        if learn_hidden:
            model.syn1neg[word_indices] += outer(gb, l1)
        neu1e += dot(gb, l2b)

    if learn_vectors:
        # update W (input -> hidden of NN)
        if not model.cbow_mean and input_word_indices:
            neu1e /= len(input_word_indices)
        for i in input_word_indices:
            model.syn0[i] += neu1e * model.syn0_lockf[i]

    return neu1e
    
class Vocab(object):
    """
    A single vocabulary item, used internally for collecting per-word frequency/sampling info,
    and for constructing binary trees (incl. both word leaves and inner nodes).
    
    """
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key])
                for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))

RULE_DEFAULT = 0
RULE_DISCARD = 1
RULE_KEEP = 2

def keep_vocab_item(word, count, min_count, trim_rule=None):
    default_res = count >= min_count

    if trim_rule is None:
        return default_res
    else:
        rule_res = trim_rule(word, count, min_count)
        if rule_res == RULE_KEEP:
            return True
        elif rule_res == RULE_DISCARD:
            return False
        else:
            return default_res

def prune_vocab(vocab, min_reduce, trim_rule=None):
    """
    Remove all entries from the `vocab` dictionary with count smaller than `min_reduce`.
    Modifies `vocab` in place, returns the sum of all counts that were pruned.
    
    """
    result = 0
    old_len = len(vocab)
    for w in list(vocab):  # make a copy of dict's keys
        if not keep_vocab_item(w, vocab[w], min_reduce, trim_rule):  # vocab[w] <= min_reduce:
            result += vocab[w]
            del vocab[w]
    logger.info("pruned out %i tokens with count <=%i (before %i, after %i)",
                old_len - len(vocab), min_reduce, old_len, len(vocab))
    return result


# matutils.zeros_aligned():
def zeros_aligned(shape, dtype, order='C', align=128):
    """Like `numpy.zeros()`, but the array will be aligned at `align` byte boundary"""
    nbytes = prod(shape, dtype=numpy.int64) * numpy.dtype(dtype).itemsize
    buffer = zeros(nbytes + align, dtype=uint8)
    start_index = -buffer.ctypes.data % align
    return buffer[start_index : start_index + nbytes].view(dtype).reshape(shape, order=order)
    
# utils.qsize():
def qsize(queue):
    """Return the (approximate) queue size where available;"""
    try:
        return queue.qsize()
    except NotImplementedError:
        # OS X doesn't support qsize
        return -1

class Word2Vec(SaveLoad):
#class Word2Vec(object):  # without save/load
    """
    """
    def __init__(
            self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH):
        """
        """
        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.index2word = []  # map from a word's matrix index (int) to word (string)
        self.sg = int(sg)
        self.cum_table = None  # for negative sampling
        self.vector_size = int(size)
        self.layer1_size = int(size)
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.min_alpha_yet_reached = float(alpha)  # To warn user if alpha increases
        self.window = int(window)
        self.max_vocab_size = max_vocab_size
        self.seed = seed
        self.random = random.RandomState(seed)
        self.min_count = min_count
        self.sample = sample
        self.workers = int(workers)
        self.min_alpha = float(min_alpha)
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.iter = iter
        self.null_word = null_word
        self.train_count = 0
        self.total_train_time = 0
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words


        if sentences is not None:
            if isinstance(sentences, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. "
                                "Try an iterator.")
            self.build_vocab(sentences, trim_rule=trim_rule)
            self.train(sentences)

    def build_vocab(self, sentences, keep_raw_vocab=False, trim_rule=None, progress_per=10000):
        """
        Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.
        
        """
        self.scan_vocab(sentences, progress_per=progress_per, trim_rule=trim_rule)  # initial survey
        self.scale_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule)  # trim by min_count & precalculate downsampling
        self.finalize_vocab()  # build tables & arrays

    def scan_vocab(self, sentences, progress_per=10000, trim_rule=None):
        """Do an initial scan of all words appearing in sentences."""
        logger.info("collecting all words and their counts")
        sentence_no = -1
        total_words = 0
        min_reduce = 1
        vocab = defaultdict(int)
        checked_string_types = 0
        for sentence_no, sentence in enumerate(sentences):
            if not checked_string_types:
                if isinstance(sentence, str):
                    logger.warn("Each 'sentence' item should be a list of words (usually unicode strings)."
                                "First item here is instead plain %s.", type(sentence))
                checked_string_types += 1
            if sentence_no % progress_per == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words, keeping %i word types",
                            sentence_no, sum(vocab.values()) + total_words, len(vocab))
            for word in sentence:
                vocab[word] += 1

            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                total_words += prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                min_reduce += 1

        total_words += sum(vocab.values())
        logger.info("collected %i word types from a corpus of %i raw words and %i sentences",
                    len(vocab), total_words, sentence_no + 1)
        self.corpus_count = sentence_no + 1
        self.raw_vocab = vocab

    def scale_vocab(self, min_count=None, sample=None, dry_run=False,
                    keep_raw_vocab=False, trim_rule=None):
        """
        """
        min_count = min_count or self.min_count
        sample = sample or self.sample

        # Discard words less-frequent than min_count
        if not dry_run:
            self.index2word = []
            # make stored settings match these applied settings
            self.min_count = min_count
            self.sample = sample
            self.vocab = {}
        drop_unique, drop_total, retain_total, original_total = 0, 0, 0, 0
        retain_words = []
        for word, v in self.raw_vocab.items():
            if keep_vocab_item(word, v, min_count, trim_rule=trim_rule):
                retain_words.append(word)
                retain_total += v
                original_total += v
                if not dry_run:
                    self.vocab[word] = Vocab(count=v, index=len(self.index2word))
                    self.index2word.append(word)
            else:
                drop_unique += 1
                drop_total += v
                original_total += v
        logger.info("min_count=%d retains %i unique words (drops %i)",
                    min_count, len(retain_words), drop_unique)
        logger.info("min_count leaves %i word corpus (%i%% of original %i)",
                    retain_total, retain_total * 100 / max(original_total, 1), original_total)

        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        elif sample < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_total
        else:
            # new shorthand: sample >= 1 means downsample all words with higher count than sample
            threshold_count = int(sample * (3 + sqrt(5)) / 2)

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = self.raw_vocab[w]
            word_probability = (sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                self.vocab[w].sample_int = int(round(word_probability * 2**32))

        if not dry_run and not keep_raw_vocab:
            logger.info("deleting the raw counts dictionary of %i items", len(self.raw_vocab))
            self.raw_vocab = defaultdict(int)

        logger.info("sample=%g downsamples %i most-common words", sample, downsample_unique)
        logger.info("downsampling leaves estimated %i word corpus (%.1f%% of prior %i)",
                    downsample_total, downsample_total * 100.0 / max(retain_total, 1), retain_total)

        # return from each step: words-affected, resulting-corpus-size
        report_values = {'drop_unique': drop_unique, 'retain_total': retain_total,
                         'downsample_unique': downsample_unique, 'downsample_total': int(downsample_total)}

        # print extra memory estimates
        report_values['memory'] = self.estimate_memory(vocab_size=len(retain_words))

        return report_values


    def finalize_vocab(self):
        """Build tables and model weights based on final vocabulary settings."""
        if not self.index2word:
            self.scale_vocab()
        if self.sorted_vocab:
            self.sort_vocab()
        if self.hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree()
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table()
        if self.null_word:
            # create null pseudo-word for padding when using concatenative L1 (run-of-words)
            # this word is only ever input - never predicted -
            # so count, huffman-point, etc donesn't matter
            word, v = '\0', Vocab(count=1, sample_int=0)
            v.index = len(self.vocab)
            self.index2word.append(word)
            self.vocab[word] = v
        # set initial/projection and hidden weights
        self.reset_weights()

    def sort_vocab(self):
        """Sort the vocabulary so the most frequent words have the lowest indexes."""
        if hasattr(self, 'syn0'):
            raise RuntimeError("must sort before initializing vectors/weights")
        self.index2word.sort(key=lambda word: self.vocab[word].count, reverse=True)
        for i, word in enumerate(self.index2word):
            self.vocab[word].index = i

    def create_binary_tree(self):
        """
        Create a binary Huffman tree using stored vocabulary word counts. Frequent words
        will have shorter binary codes. Called internally from `build_vocab()`.
        
        """
        logger.info("constructing a huffman tree from %i words", len(self.vocab))

        # build the huffman tree
        heap = list(self.vocab.values())
        heapq.heapify(heap)
        for i in range(len(self.vocab) - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, Vocab(count=min1.count + min2.count,
                                       index=i + len(self.vocab), left=min1, right=min2))

        # recurse over the tree, assigning a binary code to each vocabulary word
        if heap:
            max_depth, stack = 0, [(heap[0], [], [])]
            while stack:
                node, codes, points = stack.pop()
                if node.index < len(self.vocab):
                    # leaf node => store its path from the root
                    node.code, node.point = codes, points
                    max_depth = max(len(codes), max_depth)
                else:
                    # inner node => continue recursion
                    points = array(list(points) + [node.index - len(self.vocab)], dtype=uint32)
                    stack.append((node.left, array(list(codes) + [0], dtype=uint8), points))
                    stack.append((node.right, array(list(codes) + [1], dtype=uint8), points))

            logger.info("built huffman tree with maximum node depth %i", max_depth)

    def make_cum_table(self, power=0.75, domain=2**31 - 1):
        """
        Create a cumulative-distribution table using sorted vocabulary word counts
        for drawing random words in the negative-sampling training routines.

        To draw a word index, choose a random integer up to the maximum value in the
        table (cum_table[-1]), then finding that integer's sorted insertion point
        (as if bysect_left or ndarray.searchsorted()). That insertion point is the
        drawn index, coming up in proportion equal to the increment at that slot.

        Called internally from `build_vocab()`.
        """
        vocab_size = len(self.index2word)
        self.cum_table = zeros(vocab_size, dtype=uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = float(sum([self.vocab[word].count**power for word in self.vocab]))
        cumulative = 0.0
        for word_index in range(vocab_size):
            cumulative += self.vocab[self.index2word[word_index]].count**power / train_words_pow
            self.cum_table[word_index] = round(cumulative * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain

    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        self.syn0 = empty((len(self.vocab), self.vector_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in range(len(self.vocab)):
            # construct deterministic seed from word AND seed argument
            self.syn0[i] = self.seeded_vector(self.index2word[i] + str(self.seed))
        if self.hs:
            self.syn1 = zeros((len(self.vocab), self.layer1_size), dtype=REAL)
        if self.negative:
            self.syn1neg = zeros((len(self.vocab), self.layer1_size), dtype=REAL)
        self.syn0norm = None

        self.syn0_lockf = ones(len(self.vocab), dtype=REAL)  # zeros suppress learning

    def seeded_vector(self, seed_string):
        """Create one 'random' vector (but deterministic by seed_string)"""
        once = random.RandomState(self.hashfxn(seed_string) & 0xffffffff)
        return (once.rand(self.vector_size) - 0.5) / self.vector_size

    def estimate_memory(self, vocab_size=None, report=None):
        """Estimate required memory for a model using current settings and provided vocabulary size"""
        vocab_size = vocab_size or len(self.vocab)
        report = report or {}
        report['vocab'] = vocab_size * (700 if self.hs else 500)
        report['syn0'] = vocab_size * self.vector_size * dtype(REAL).itemsize
        if self.hs:
            report['syn1'] = vocab_size * self.layer1_size * dtype(REAL).itemsize
        if self.negative:
            report['syn1neg'] = vocab_size * self.layer1_size * dtype(REAL).itemsize
        report['total'] = sum(report.values())
        logger.info("estimated required memory for %i words and %i dimensions: %i bytes",
                    vocab_size, self.vector_size, report['total'])
        return report

    def _do_train_job(self, sentences, alpha, inits):
        """
        Train a single batch of sentences. Return 2-tuple
        `(effective word count after ignoring unknown words and sentence length trimming,
        total word count)`.
        """
        work, neu1 = inits
        tally = 0
        if self.sg:
            tally += train_batch_sg(self, sentences, alpha, work)
        else:
            tally += train_batch_cbow(self, sentences, alpha, work, neu1)
        return tally, self._raw_word_count(sentences)
    
    def _raw_word_count(self, job):
        """Return the number of words in a given job."""
        return sum(len(sentence) for sentence in job)
    
    def train(self, sentences, total_words=None, word_count=0,
              total_examples=None, queue_factor=2, report_delay=1.0):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        For Word2Vec, each sentence must be a list of unicode strings. (Subclasses may accept other examples)

        To support linear learning-rate decay from (initial) alpha to min_alpha, either
        total_examples (count of sentences) or total_words (count of raw words in sentences)
        should be provided, unless the sentences are the same as those that were used to
        initially build the vocabulary.
        
        """
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("C extension not loaded for Word2Vec, training will be slow. "
                          "Install a C compiler and reinstall gensim for fast training.")
            self.neg_labels = []
            if self.negative > 0:
                # precompute negative labels optimization for pure-python training
                self.neg_labels = zeros(self.negative + 1)
                self.neg_labels[0] = 1.

        logger.info(
            "training model with %i workers on %i vocabulary and %i features, "
            "using sg=%s hs=%s sample=%s negative=%s",
            self.workers, len(self.vocab), self.layer1_size, self.sg,
            self.hs, self.sample, self.negative)

        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")
        if not hasattr(self, 'syn0'):
            raise RuntimeError("you must first finalize vocabulary before training the model")

        if total_words is None and total_examples is None:
            if self.corpus_count:
                total_examples = self.corpus_count
                logger.info("expecting %i sentences, matching count from corpus used for "
                            "vocabulary survey", total_examples)
            else:
                raise ValueError("you must provide either total_words or total_examples, "
                                 "to enable alpha and progress calculations")

        job_tally = 0

        if self.iter > 1:
            sentences = RepeatCorpusNTimes(sentences, self.iter)
            total_words = total_words and total_words * self.iter
            total_examples = total_examples and total_examples * self.iter

        def worker_loop():
            """Train the model, lifting lists of sentences from the job_queue."""
            work = zeros_aligned(self.layer1_size, dtype=REAL)  # per-thread private work memory
            neu1 = zeros_aligned(self.layer1_size, dtype=REAL)
            jobs_processed = 0
            while True:
                job = job_queue.get()
                if job is None:
                    progress_queue.put(None)
                    break  # no more jobs => quit this worker
                sentences, alpha = job
                tally, raw_tally = self._do_train_job(sentences, alpha, (work, neu1))
                progress_queue.put((len(sentences), tally, raw_tally))  # report back progress
                jobs_processed += 1
            logger.debug("worker exiting, processed %i jobs", jobs_processed)

        def job_producer():
            """Fill jobs queue using the input `sentences` iterator."""
            job_batch, batch_size = [], 0
            pushed_words, pushed_examples = 0, 0
            next_alpha = self.alpha
            if next_alpha > self.min_alpha_yet_reached:
                logger.warn("Effective 'alpha' higher than previous training cycles")
            self.min_alpha_yet_reached = next_alpha
            job_no = 0

            for sent_idx, sentence in enumerate(sentences):
                sentence_length = self._raw_word_count([sentence])

                # can we fit this sentence into the existing job batch?
                if batch_size + sentence_length <= self.batch_words:
                    # yes => add it to the current job
                    job_batch.append(sentence)
                    batch_size += sentence_length
                else:
                    # no => submit the existing job
                    logger.debug(
                        "queueing job #%i (%i words, %i sentences) at alpha %.05f",
                        job_no, batch_size, len(job_batch), next_alpha)
                    job_no += 1
                    job_queue.put((job_batch, next_alpha))

                    # update the learning rate for the next job
                    if self.min_alpha < next_alpha:
                        if total_examples:
                            # examples-based decay
                            pushed_examples += len(job_batch)
                            progress = 1.0 * pushed_examples / total_examples
                        else:
                            # words-based decay
                            pushed_words += self._raw_word_count(job_batch)
                            progress = 1.0 * pushed_words / total_words
                        next_alpha = self.alpha - (self.alpha - self.min_alpha) * progress
                        next_alpha = max(self.min_alpha, next_alpha)

                    # add the sentence that didn't fit as the first item of a new job
                    job_batch, batch_size = [sentence], sentence_length

            # add the last job too (may be significantly smaller than batch_words)
            if job_batch:
                logger.debug(
                    "queueing job #%i (%i words, %i sentences) at alpha %.05f",
                    job_no, batch_size, len(job_batch), next_alpha)
                job_no += 1
                job_queue.put((job_batch, next_alpha))

            if job_no == 0 and self.train_count == 0:
                logger.warning(
                    "train() called with an empty iterator (if not intended, "
                    "be sure to provide a corpus that offers restartable "
                    "iteration = an iterable).")

            # give the workers heads up that they can finish -- no more work!
            for _ in range(self.workers):
                job_queue.put(None)
            logger.debug("job loop exiting, total %i jobs", job_no)

        # buffer ahead only a limited number of jobs..
        # this is the reason we can't simply use ThreadPool :(
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in range(self.workers)]
        unfinished_worker_count = len(workers)
        workers.append(threading.Thread(target=job_producer))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        example_count, trained_word_count, raw_word_count = 0, 0, word_count
        start, next_report = default_timer() - 0.00001, 1.0

        while unfinished_worker_count > 0:
            report = progress_queue.get()  # blocks if workers too slow
            if report is None:  # a thread reporting that it finished
                unfinished_worker_count -= 1
                logger.info("worker thread finished; awaiting finish of %i more threads",
                            unfinished_worker_count)
                continue
            examples, trained_words, raw_words = report
            job_tally += 1

            # update progress stats
            example_count += examples
            trained_word_count += trained_words  # only words in vocab & sampled
            raw_word_count += raw_words

            # log progress once every report_delay seconds
            elapsed = default_timer() - start
            if elapsed >= next_report:
                if total_examples:
                    # examples-based progress %
                    logger.info(
                        "PROGRESS: at %.2f%% examples, %.0f words/s, in_qsize %i, out_qsize %i",
                        100.0 * example_count / total_examples, trained_word_count / elapsed,
                        qsize(job_queue), qsize(progress_queue))
                else:
                    # words-based progress %
                    logger.info(
                        "PROGRESS: at %.2f%% words, %.0f words/s, in_qsize %i, out_qsize %i",
                        100.0 * raw_word_count / total_words, trained_word_count / elapsed,
                        qsize(job_queue), qsize(progress_queue))
                next_report = elapsed + report_delay

        # all done; report the final stats
        elapsed = default_timer() - start
        logger.info(
            "training on %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            raw_word_count, trained_word_count, elapsed, trained_word_count / elapsed)
        if job_tally < 10 * self.workers:
            logger.warn("under 10 jobs per worker: "
                        "consider setting a smaller `batch_words` for smoother alpha decay")

        # check that the input corpus hasn't changed during iteration
        if total_examples and total_examples != example_count:
            logger.warn("supplied example count (%i) did not equal expected count (%i)",
                        example_count, total_examples)
        if total_words and total_words != raw_word_count:
            logger.warn("supplied raw word count (%i) did not equal expected count (%i)",
                        raw_word_count, total_words)

        self.train_count += 1  # number of times train() has been called
        self.total_train_time += elapsed
        self.clear_sims()
        return trained_word_count

    def clear_sims(self):
        self.syn0norm = None

    def init_sims(self, replace=False):
        if getattr(self, 'syn0norm', None) is None or replace:
            logger.info("precomputing L2-norms of word weight vectors")
            if replace:
                for i in range(self.syn0.shape[0]):
                    self.syn0[i, :] /= sqrt((self.syn0[i, :]**2).sum(-1))
                self.syn0norm = self.syn0
                if hasattr(self, 'syn1'):
                    del self.syn1
            else:
                self.syn0norm = (self.syn0 / sqrt((self.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)

    def save_word2vec_format(self, fname, fvocab=None, binary=False):
        """
        """
        if fvocab is not None:
            logger.info("storing vocabulary in %s" % (fvocab))
            with smart_open(fvocab, 'wb') as vout:
                for word, vocab in sorted(self.vocab.items(), key=lambda item: -item[1].count):
                    vout.write(to_utf8("%s %s\n" % (word, vocab.count)))
        logger.info("storing %sx%s projection weights into %s" % (
            len(self.vocab), self.vector_size, fname))
        assert (len(self.vocab), self.vector_size) == self.syn0.shape
        with smart_open(fname, 'wb') as fout:
            fout.write(to_utf8("%s %s\n" % self.syn0.shape))
            for word, vocab in sorted(self.vocab.items(), key=lambda item: -item[1].count):
                row = self.syn0[vocab.index]
                if binary:
                    fout.write(to_utf8(word) + b" " + row.tobytes())
                else:
                    fout.write(to_utf8("%s %s\n" % (
                        word, ' '.join("%f" % val for val in row))))

    @classmethod
    def load_word2vec_format(cls, fname, fvocab=None, binary=False, encoding='utf-8',
                             unicode_errors='strict', limit=None, datatype=REAL):
        """
        """
        counts = None
        if fvocab is not None:
            logger.info("loading word counts from %s", fvocab)
            counts = {}
            with smart_open(fvocab) as fin:
                for line in fin:
                    word, count = to_unicode(line).strip().split()
                    counts[word] = int(count)

        logger.info("loading projection weights from %s", fname)
        with smart_open(fname) as fin:
            header = to_unicode(fin.readline(), encoding=encoding)
            vocab_size, vector_size = map(int, header.split())
            if limit:
                vocab_size = min(vocab_size, limit)
            result = cls(size=vector_size)
            result.syn0 = zeros((vocab_size, vector_size), dtype=datatype)

            def add_word(word, weights):
                word_id = len(result.vocab)
                if word in result.vocab:
                    logger.warning("duplicate word '%s' in %s, ignoring all but first", word, fname)
                    return
                if counts is None:
                    result.vocab[word] = Vocab(index=word_id, count=vocab_size - word_id)
                elif word in counts:
                    result.vocab[word] = Vocab(index=word_id, count=counts[word])
                else:
                    logger.warning("vocabulary file is incomplete: '%s' is missing", word)
                    result.vocab[word] = Vocab(index=word_id, count=None)
                result.syn0[word_id] = weights
                result.index2word.append(word)

            if binary:
                binary_len = dtype(REAL).itemsize * vector_size
                for line_no in range(vocab_size):
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == b' ':
                            break
                        if ch == b'':
                            raise EOFError("unexpected end of input; "
                                           "is count incorrect or file otherwise damaged?")
                        if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                            word.append(ch)
                    word = to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                    weights = fromstring(fin.read(binary_len), dtype=REAL)
                    add_word(word, weights)
            else:
                for line_no in range(vocab_size):
                    line = fin.readline()
                    if line == b'':
                        raise EOFError("unexpected end of input; "
                                       "is count incorrect or file otherwise damaged")
                    parts = to_unicode(line.rstrip(), encoding=encoding,
                                       errors=unicode_errors).split(" ")
                    if len(parts) != vector_size + 1:
                        raise ValueError("invalid vector on line %s "
                                         "(is this really the text format?)" % (line_no))
                    word, weights = parts[0], list(map(REAL, parts[1:]))
                    add_word(word, weights)

        if result.syn0.shape[0] != len(result.vocab):
            logger.info(
                "duplicate words detected, shrinking matrix size from %i to %i",
                result.syn0.shape[0], len(result.vocab))
            result.syn0 = ascontiguousarray(result.syn0[: len(result.vocab)])
        assert (len(result.vocab), result.vector_size) == result.syn0.shape

        logger.info("loaded %s matrix from %s" % (result.syn0.shape, fname))
        return result


    def __str__(self):
        return "%s(vocab=%s, size=%s, alpha=%s)" % (
            self.__class__.__name__, len(self.index2word), self.vector_size, self.alpha)
                    
    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors, recalculable table
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'table', 'cum_table'])
        super(Word2Vec, self).save(*args, **kwargs)  # utils.SaveLoad.save()

    save.__doc__ = SaveLoad.save.__doc__

    @classmethod
    def load(cls, *args, **kwargs):
        model = super(Word2Vec, cls).load(*args, **kwargs)
        # update older models
        if hasattr(model, 'table'):
            delattr(model, 'table')
        if model.negative and hasattr(model, 'index2word'):
            model.make_cum_table()
        if not hasattr(model, 'corpus_count'):
            model.corpus_count = None
        for v in model.vocab.values():
            if hasattr(v, 'sample_int'):
                break
            elif hasattr(v, 'sample_probability'):
                v.sample_int = int(round(v.sample_probability * 2**32))
                del v.sample_probability
        if not hasattr(model, 'syn0_lockf') and hasattr(model, 'syn0'):
            model.syn0_lockf = ones(len(model.syn0), dtype=REAL)
        if not hasattr(model, 'random'):
            model.random = random.RandomState(model.seed)
        if not hasattr(model, 'train_count'):
            model.train_count = 0
            model.total_train_time = 0
        return model

    

    def similarity(self, w1, w2):
        return dot(unitvec(self[w1]), unitvec(self[w2]))

    def most_similar(self, positive=[], negative=[], topn=10,
                     restrict_vocab=None, indexer=None):
        """
        """
        self.init_sims()

        if isinstance(positive, str) and not negative:
            positive = [positive]

        positive = [(word, 1.0) if isinstance(word, string_types + (ndarray,)) else word
                    for word in positive]
        negative = [(word, -1.0) if isinstance(word, string_types + (ndarray,)) else word
                    for word in negative]

        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            elif word in self.vocab:
                mean.append(weight * self.syn0norm[self.vocab[word].index])
                all_words.add(self.vocab[word].index)
            else:
                raise KeyError("word '%s' not in vocabulary" % word)

        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = unitvec(array(mean).mean(axis=0)).astype(REAL)

        if indexer is not None:
            return indexer.most_similar(mean, topn)

        limited = self.syn0norm if restrict_vocab is None else self.syn0norm[:restrict_vocab]
        dists = dot(limited, mean)
        if not topn:
            return dists
        best = argsort(dists, topn=topn + len(all_words), reverse=True)

        result = [(self.index2word[sim], float(dists[sim]))
                  for sim in best if sim not in all_words]
        return result[:topn]
                
    def accuracy(self, questions, restrict_vocab=30000,
                 most_similar=most_similar, case_insensitive=True):
        """
        """
        pass
                    



    

class RepeatCorpusNTimes(SaveLoad):
# class RepeatCorpusNTimes(object):
    def __init__(self, corpus, n):
        """
        Repeat a `corpus` `n` times.

        >>> corpus = [[(1, 0.5)], []]
        >>> list(RepeatCorpusNTimes(corpus, 3))  # repeat 3 times
        [[(1, 0.5)], [], [(1, 0.5)], [], [(1, 0.5)], []]
        """
        self.corpus = corpus
        self.n = n

    def __iter__(self):
        for _ in range(self.n):
            for document in self.corpus:
                yield document



def any2utf8(text, errors='strict', encoding='utf-8'):
    if isinstance(text, str):
        return text.encode('utf8')
    return str(text, encoding, errors=errors).encode('utf8')
to_utf8 = any2utf8
            
def any2unicode(text, encoding='utf-8', errors='strict'):
    if isinstance(text, str):
        return text
    return str(text, encoding, errors=errors)
to_unicode = any2unicode


class LineSentence(object):
    """
    Simple format: one sentence = one line; words already preprocessed and separated by whitespace.
    """

    def __init__(self, source, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None):
        """
        `source` can be either a string or a file object.
        Clip the file to the first `limit` lines (or no clipped if limit is None, the default).

        Example:

            sentences = LineSentence('myfile.txt')

        Or for compressed files::

            sentence = LineSentence('compressed_text.txt.bz2')
            sentence = LineSentence('compressed_text.txt.gz')
        
        """
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit

    def __iter__(self):
        """Iterate through the lines in the source"""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for line in itertools.islice(self.source, self.limit):
                line = to_unicode(line).split()
                i = 0
                while i < len(line):
                    yield line[i : i + self.max_sentence_length]
                    i += self.max_sentence_length
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with smart_open(self.source) as fin:
                for line in itertools.islice(fin, self.limit):
                    line = to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i : i + self.max_sentence_length]
                        i += self.max_sentence_length





if __name__ == '__main__':
    import argparse
    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO)  # DEBUG => INFO
    logging.info("running %s", " ".join(sys.argv))
    logging.info("using optimization %s", FAST_VERSION)

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
#        print(globals()['__doc__'] % locals())  ???
        sys.exit(1)

    from w2v_r1 import Word2Vec  # avoid referencing __main__ in pickle

    seterr(all='raise')  # don't ignore numpy errors

    parser = argparse.ArgumentParser()
    parser.add_argument("-train", help="Use text data from file TRAIN to train the model", required=True) 
    parser.add_argument("-output", help="Use file OUTPUT to save the resulting word vectors") 
    parser.add_argument("-window", help="Set max skip length WINDOW between words; default is 5", type=int, default=5) 
    parser.add_argument("-size", help="Set size of word vectors; default is 100", type=int, default=100) 
    parser.add_argument("-sample", help="Set threshold for occurrence of words. Those that appear with higher frequency in the training data will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)", type=float, default=1e-3) 
    parser.add_argument("-hs", help="Use Hierarchical Softmax; default is 0 (not used)", type=int, default=0, choices=[0, 1]) 
    parser.add_argument("-negative", help="Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)", type=int, default=5) 
    parser.add_argument("-threads", help="Use THREADS threads (default 12)", type=int, default=12) 
    parser.add_argument("-iter", help="Run more training iterations (default 5)", type=int, default=5) 
    parser.add_argument("-min_count", help="This will discard words that appear less than MIN_COUNT times; default is 5", type=int, default=5) 
    parser.add_argument("-cbow", help="Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)", type=int, default=1, choices=[0, 1]) 
    parser.add_argument("-binary", help="Save the resulting vectors in binary mode; default is 0 (off)", type=int, default=0, choices=[0, 1]) 
    parser.add_argument("-accuracy", help="Use questions from file ACCURACY to evaluate the model")

    args = parser.parse_args()

    if args.cbow == 0:
        skipgram = 1
    else:
        skipgram = 0

    corpus = LineSentence(args.train)

    model = Word2Vec(
        corpus, size=args.size, min_count=args.min_count, workers=args.threads,
        window=args.window, sample=args.sample, sg=skipgram, hs=args.hs,
        negative=args.negative, cbow_mean=1, iter=args.iter)

    if args.output:
        outfile = args.output
        model.save_word2vec_format(outfile, binary=args.binary)
    else:
        outfile = args.train
        model.save(outfile + '.model')
    if args.binary == 1:
        model.save_word2vec_format(outfile + '.model.bin', binary=True)
    else:
        model.save_word2vec_format(outfile + '.model.txt', binary=False)

    if args.accuracy:
        model.accuracy(args.accuracy)

#    model = Word2Vec.load(fname)
#    model = Word2Vec.load_word2vec_format('./vectors.txt', binary=False)  # C text format
#    model = Word2Vec.load_word2vec_format('./vectors.bin', binary=True)  # C binary format
# 
#    model.most_similar(positive=['woman', 'king'], negative=['man'])
#    model.doesnt_match("breakfast sereal dinner lunch".split())
#    model.similarity('woman', 'man')
#    model['computer']  # raw numpy vector of a word
# 
#    # If you're finished training a model (= no more updates, only querying), you can do
#    model.init_sims(replace=True)

    logger.info("finished running %s", program)





# ================================================================================

# utils.SaveLoad():
class SaveLoad(object):
    """
    Objects which inherit from this class have save/load functions,
    which un/pickle them to disk.

    This uses pickle for de/serializing, so objects must not contain
    unpicklable attributes, such as lambda functions etc.
    
    """
    @classmethod
    def load(cls, fname, mmap=None):
        """
        """
        logger.info("loading %s object from %s" % (cls.__name__, fname))

        compress, subname = SaveLoad._adapt_by_suffix(fname)

        obj = unpickle(fname)
        obj._load_specials(fname, mmap, compress, subname)
        return obj

    def _load_specials(self, fname, mmap, compress, subname):
        """
        """

        mmap_error = lambda x, y: IOError(
            'Cannot mmap compressed object %s in file %s. ' % (x, y) +
            'Use `load(fname, mmap=None)` or uncompress files manually.')

        for attrib in getattr(self, '__recursive_saveloads', []):
            cfname = '.'.join((fname, attrib))
            logger.info("loading %s recursively from %s.* with mmap=%s" % (
                attrib, cfname, mmap))
            getattr(self, attrib)._load_specials(cfname, mmap, compress, subname)

        for attrib in getattr(self, '__numpys', []):
            logger.info("loading %s from %s with mmap=%s" % (
                attrib, subname(fname, attrib), mmap))

            if compress:
                if mmap:
                    raise mmap_error(attrib, subname(fname, attrib))

                val = numpy.load(subname(fname, attrib))['val']
            else:
                val = numpy.load(subname(fname, attrib), mmap_mode=mmap)

            setattr(self, attrib, val)

        for attrib in getattr(self, '__scipys', []):
            logger.info("loading %s from %s with mmap=%s" % (
                attrib, subname(fname, attrib), mmap))
            sparse = unpickle(subname(fname, attrib))
            if compress:
                if mmap:
                    raise mmap_error(attrib, subname(fname, attrib))

                with numpy.load(subname(fname, attrib, 'sparse')) as f:
                    sparse.data = f['data']
                    sparse.indptr = f['indptr']
                    sparse.indices = f['indices']
            else:
                sparse.data = numpy.load(subname(fname, attrib, 'data'), mmap_mode=mmap)
                sparse.indptr = numpy.load(subname(fname, attrib, 'indptr'), mmap_mode=mmap)
                sparse.indices = numpy.load(subname(fname, attrib, 'indices'), mmap_mode=mmap)

            setattr(self, attrib, sparse)

        for attrib in getattr(self, '__ignoreds', []):
            logger.info("setting ignored attribute %s to None" % (attrib))
            setattr(self, attrib, None)
            

    @staticmethod
    def _adapt_by_suffix(fname):
        """Give appropriate compress setting and filename formula"""
        if fname.endswith('.gz') or fname.endswith('.bz2'):
            compress = True
            subname = lambda *args: '.'.join(list(args) + ['npz'])
        else:
            compress = False
            subname = lambda *args: '.'.join(list(args) + ['npy'])
        return (compress, subname)



