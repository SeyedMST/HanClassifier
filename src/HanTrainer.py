from __future__ import print_function
import argparse
import os
import sys
import time
import re
import tensorflow as tf
import numpy as np


from .vocab_utils import Vocab
from .HanDataStream import HanDataStream
from .HanModelGraph import HanModelGraph
from .namespace_utils import save_namespace

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

def evaluate(dataStream, valid_graph, sess, outpath=None, label_vocab=None, mode = None):
    if outpath is not None: outfile = open(outpath, 'wt')
    total_tags = 0.0
    correct_tags = 0.0
    dataStream.reset()
    num_classes = label_vocab.size ()
    confusion_matrix = np.zeros((num_classes, num_classes))
    for instance_index in range(dataStream.get_num_instance()):
        cur_instance = dataStream.get_instance(instance_index)
        (label, text, label_id, word_idx, sents_length) = cur_instance
        feed_dict = {
                    valid_graph.truth: np.array([label_id]),
                    valid_graph.sents_length: sents_length,
                    valid_graph.in_text_words: word_idx,
                }
        if mode != None: # we just want to detect the topic of a single input text.
            return label_vocab.getWord(sess.run (valid_graph.predictions, feed_dict=feed_dict) [0])
        total_tags += 1 #each test instance has 1 instance per batch
        correct_tags += sess.run(valid_graph.eval_correct, feed_dict=feed_dict)
        if outpath is not None:
            prediction = sess.run(valid_graph.predictions, feed_dict=feed_dict)[0]
            confusion_matrix [prediction, label_id] += 1
            predicted_label = label_vocab.getWord(prediction)
            if predicted_label != label:
                outline = predicted_label +"\n"+ text + "__label__" +label +"\n\n"
                outfile.write(outline)
            # if (label, predicted_label) not in confusion_matrix:
            #     confusion_matrix [(label, predicted_label)] = 0
            # confusion_matrix [(label, predicted_label)] += 1
    accuracy = correct_tags / total_tags * 100
    if outpath is not None:
        #outfile.write(str(confusion_matrix))
        outfile.close ()
        return accuracy, confusion_matrix
    return accuracy

def main(_):
    print('Configurations:')
    print(FLAGS)
    train_path = FLAGS.train_path
    dev_path = FLAGS.dev_path
    test_path = FLAGS.test_path
    word_vec_path = FLAGS.word_vec_path
    log_dir = FLAGS.model_dir
    result_dir = '../result'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    path_prefix = log_dir + "/Han.{}".format(FLAGS.suffix)
    save_namespace(FLAGS, path_prefix + ".config.json")
    word_vocab = Vocab(word_vec_path, fileformat='txt3')
    best_path = path_prefix + '.best.model'
    label_path = path_prefix + ".label_vocab"

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
    trainDataStream = HanDataStream(inpath=train_path, word_vocab=word_vocab,
                                              label_vocab=label_vocab,
                                              isShuffle=True, isLoop=True,
                                              max_sent_length=FLAGS.max_sent_length)
    devDataStream = HanDataStream(inpath=dev_path, word_vocab=word_vocab,
                                              label_vocab=label_vocab,
                                              isShuffle=False, isLoop=True,
                                              max_sent_length=FLAGS.max_sent_length)
    testDataStream = HanDataStream(inpath=test_path, word_vocab=word_vocab,
                                              label_vocab=label_vocab,
                                              isShuffle=False, isLoop=True,
                                              max_sent_length=FLAGS.max_sent_length)

    print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    print('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))

    with tf.Graph().as_default():
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            train_graph = HanModelGraph(num_classes=num_classes, word_vocab=word_vocab,
                                                  dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate,
                                                  optimize_type=FLAGS.optimize_type,
                                                  lambda_l2=FLAGS.lambda_l2,
                                                  context_lstm_dim=FLAGS.context_lstm_dim,
                                                  is_training=True, batch_size = FLAGS.batch_size)
            tf.summary.scalar("Training Loss", train_graph.loss)  # Add a scalar summary for the snapshot loss.
        print("Train Graph Build")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            valid_graph = HanModelGraph(num_classes=num_classes, word_vocab=word_vocab,
                                                  dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate,
                                                  optimize_type=FLAGS.optimize_type,
                                                  lambda_l2=FLAGS.lambda_l2,
                                                  context_lstm_dim=FLAGS.context_lstm_dim,
                                                  is_training=False, batch_size = 1)
        print ("dev Graph Build")
        initializer = tf.global_variables_initializer()
        vars_ = {}
        for var in tf.all_variables():
            if "word_embedding" in var.name: continue
            #             if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        output_res_file = open(result_dir + '/' + FLAGS.suffix, 'wt')
        output_res_file.write(str(FLAGS))
        with tf.Session() as sess:
            sess.run(initializer)
            train_size = trainDataStream.get_num_instance()
            max_steps = (train_size * FLAGS.max_epochs) // FLAGS.batch_size
            epoch_size = max_steps // (FLAGS.max_epochs)  # + 1

            total_loss = 0.0
            start_time = time.time()
            best_accuracy = 0
            for step in range(max_steps):
                # read data
                _truth = []
                _sents_length = []
                _in_text_words = []
                for i in range(FLAGS.batch_size):
                    cur_instance, instance_index = trainDataStream.nextInstance ()
                    (label,text,label_id, word_idx, sents_length) = cur_instance

                    _truth.append(label_id)
                    _sents_length.append(sents_length)
                    _in_text_words.append(word_idx)

                feed_dict = {
                    train_graph.truth: np.array(_truth),
                    train_graph.sents_length: tuple(_sents_length),
                    train_graph.in_text_words: tuple(_in_text_words),
                }

                _, loss_value, _score = sess.run([train_graph.train_op, train_graph.loss
                                                     , train_graph.batch_class_scores],
                                                 feed_dict=feed_dict)
                total_loss += loss_value

                if step % 100 == 0:
                    print('{} '.format(step), end="")
                    sys.stdout.flush()
                if (step + 1) % epoch_size == 0 or (step + 1) == max_steps:
                    # print(total_loss)
                    duration = time.time() - start_time
                    start_time = time.time()
                    print(duration, step, "Loss: ", total_loss)
                    output_res_file.write('\nStep %d: loss = %.2f (%.3f sec)\n' % (step, total_loss, duration))
                    total_loss = 0.0
                    # Evaluate against the validation set.
                    output_res_file.write('valid- ')
                    dev_accuracy = evaluate(devDataStream, valid_graph, sess)
                    output_res_file.write("%.2f\n" % dev_accuracy)
                    print("Current dev accuracy is %.2f" % dev_accuracy)
                    if dev_accuracy > best_accuracy:
                        best_accuracy = dev_accuracy
                        saver.save(sess, best_path)
                    output_res_file.write('test- ')
                    test_accuracy = evaluate(testDataStream, valid_graph, sess)
                    print("Current test accuracy is %.2f" % test_accuracy)
                    output_res_file.write("%.2f\n" % test_accuracy)

    output_res_file.close()
    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str,default = '../data/news/train1000-41.txt', help='Path to the train set.')
    parser.add_argument('--dev_path', type=str, default = '../data/news/dev1000-41.txt', help='Path to the dev set.')
    parser.add_argument('--test_path', type=str, default = '../data/news/test1000-41.txt',help='Path to the test set.')
    parser.add_argument('--word_vec_path', type=str, default='../data/glove/my_wiki.fa.vec', help='Path the to pre-trained word vector model.')
    parser.add_argument('--model_dir', type=str,default = '../models',help='Directory to save model files.')
    parser.add_argument('--batch_size', type=int, default= 5, help='Number of instances in each batch.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--lambda_l2', type=float, default=0.0, help='The coefficient of L2 regularizer.')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout ratio.')
    parser.add_argument('--max_epochs', type=int, default=20, help='Maximum epochs for training.')
    parser.add_argument('--optimize_type', type=str, default='adam', help='Optimizer type.')
    parser.add_argument('--context_lstm_dim', type=int, default=100, help='Number of dimension for context representation layer.')
    parser.add_argument('--max_sent_length', type=int, default=100, help='Maximum number of words within each sentence.')
    parser.add_argument('--suffix', type=str, default='test41', required=False, help='Suffix of the model name.')

#     print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    sys.stdout.flush()
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)




