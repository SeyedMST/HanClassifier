from __future__ import print_function
import argparse
import os
import sys
import time
import re
import tensorflow as tf
import random
import numpy as np


from vocab_utils import Vocab
from SentenceMatchDataStream import SentenceMatchDataStream
from SentenceMatchModelGraph import SentenceMatchModelGraph
import namespace_utils

eps = 1e-8
FLAGS = None


def collect_vocabs(train_path):
    all_labels = set()
    all_words = set()
    infile = open(train_path, 'rt')
    for line in infile:
        if sys.version_info[0] < 3:
            line = line.decode('utf-8').strip()
        else:
            line = line.strip()
        items = re.split("__label__", line)
        label = items[1]
        text = re.split("\\s+",items[0])
        all_labels.add(label)
        all_words.update(text)
    infile.close()
    return (all_words, all_labels)


def main(_):
    print('Configurations:')
    print(FLAGS)

    train_path = FLAGS.train_path
    dev_path = FLAGS.dev_path
    test_path = FLAGS.test_path
    word_vec_path = FLAGS.word_vec_path
    log_dir = FLAGS.model_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    path_prefix = log_dir + "/SentenceMatch.{}".format(FLAGS.suffix)
    namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")


    word_vocab = Vocab(word_vec_path, fileformat='txt3')
    best_path = path_prefix + '.best.model'
    label_path = path_prefix + ".label_vocab"
    has_pre_trained_model = False


    print('Collect words and labels ...')
    (all_words, all_labels) = collect_vocabs(train_path)
    print('Number of words: {}'.format(len(all_words)))
    print('Number of labels: {}'.format(len(all_labels)))
    label_vocab = Vocab(fileformat='voc', voc=all_labels, dim=2)
    label_vocab.dump_to_txt2(label_path)

    print('word_vocab shape is {}'.format(word_vocab.word_vecs.shape))
    print('tag_vocab shape is {}'.format(label_vocab.word_vecs.shape))
    num_classes = label_vocab.size()

    print('Build HanDataStream ... ')
    trainDataStream = SentenceMatchDataStream(inpath=train_path, word_vocab=word_vocab,
                                              label_vocab=label_vocab,
                                              isShuffle=True, isLoop=True,
                                              max_sent_length=FLAGS.max_sent_length)

    devDataStream = SentenceMatchDataStream(inpath=train_path, word_vocab=word_vocab,
                                              label_vocab=label_vocab,
                                              isShuffle=False, isLoop=True,
                                              max_sent_length=FLAGS.max_sent_length)

    testDataStream = SentenceMatchDataStream(inpath=train_path, word_vocab=word_vocab,
                                              label_vocab=label_vocab,
                                              isShuffle=False, isLoop=True,
                                              max_sent_length=FLAGS.max_sent_length)

    print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    print('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))


    sys.stdout.flush()







