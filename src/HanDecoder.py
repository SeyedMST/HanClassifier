from __future__ import print_function
import argparse
from vocab_utils import Vocab
import namespace_utils

import tensorflow as tf
import HanTrainer
from HanModelGraph import  HanModelGraph
from HanDataStream import HanDataStream


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='test', help='Prefix to the models.')
    parser.add_argument('--model_dir', type=str,default = '../models',help='Directory to save model files.')
    parser.add_argument('--in_path', type=str, default='../data/news/test10-6.txt', help='the path to the test file.')
    parser.add_argument('--out_path', type=str, required=True, help='The path to the output file.')
    parser.add_argument('--word_vec_path', type=str, default='../data/glove/my_wiki.fa.vec', help='word embedding file for the input file.')

    args, unparsed = parser.parse_known_args()

    model_prefix = args.model_dir + "/Han.{}".format(args.suffix)
    in_path = args.in_path
    out_path = args.out_path + 'S'
    word_vec_path = args.word_vec_path

    # load the configuration file
    print('Loading configurations.')
    FLAGS = namespace_utils.load_namespace(model_prefix + ".config.json")
    print(FLAGS)

    # load vocabs
    print('Loading vocabs.')
    word_vocab = Vocab(word_vec_path, fileformat='txt3')
    label_vocab = Vocab(model_prefix + ".label_vocab", fileformat='txt2')
    print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
    print('label_vocab: {}'.format(label_vocab.word_vecs.shape))
    num_classes = label_vocab.size()

    testDataStream = HanDataStream(inpath=in_path, word_vocab=word_vocab,
                                              label_vocab=label_vocab,
                                              isShuffle=False, isLoop=True,
                                              max_sent_length=FLAGS.max_sent_length)

    print('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))

    best_path = model_prefix + ".best.model"
    print('Decoding on the test set:')
    with tf.Graph().as_default():
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
            valid_graph = HanModelGraph(num_classes=num_classes, word_vocab=word_vocab,
                                                  dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate,
                                                  optimize_type=FLAGS.optimize_type,
                                                  lambda_l2=FLAGS.lambda_l2,
                                                  context_lstm_dim=FLAGS.context_lstm_dim,
                                                  is_training=False, batch_size = 1)
        vars_ = {}
        for var in tf.all_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        step = 0
        saver.restore(sess, best_path)

        accuracy = HanTrainer.evaluate(testDataStream, valid_graph, sess, outpath=out_path, label_vocab=label_vocab)
        print("Accuracy for test set is %.2f" % accuracy)
