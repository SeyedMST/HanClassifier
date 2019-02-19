from __future__ import print_function
import argparse
import tensorflow as tf
import time


from .vocab_utils import Vocab
from .namespace_utils import load_namespace
from .HanTrainer import evaluate
from .HanModelGraph import  HanModelGraph
from .HanDataStream import HanDataStream

'''load word vocab and FLAGS(configuration file)'''
def load_word_vocab (word_vec_path):
    word_vocab= Vocab(word_vec_path, fileformat='txt3')
    #print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
    return word_vocab

def load_model (model_prefix, word_vocab):
    #print('Loading configurations.')
    FLAGS = load_namespace(model_prefix + ".config.json")
    label_vocab = Vocab(model_prefix + ".label_vocab", fileformat='txt2')
    #print('label_vocab: {}'.format(label_vocab.word_vecs.shape))
    num_classes = label_vocab.size()
    best_path = model_prefix + ".best.model"
    #print('Decoding on the test set:')
    with tf.Graph().as_default():
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
            valid_graph = HanModelGraph(num_classes=num_classes, word_vocab=word_vocab,
                                        dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate,
                                        optimize_type=FLAGS.optimize_type,
                                        lambda_l2=FLAGS.lambda_l2,
                                        context_lstm_dim=FLAGS.context_lstm_dim,
                                        is_training=False, batch_size=1)
        vars_ = {}
        for var in tf.all_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, best_path)
        return valid_graph, sess, label_vocab, FLAGS
# def load_FLAGS (model_prefix):
#     print('Loading configurations.')
#     FLAGS = load_namespace(model_prefix + ".config.json")
#     print(FLAGS)

def load_data_stream (in_path, word_vocab, label_vocab, FLAGS, is_inpath_input_text = False):
    testDataStream = HanDataStream(inpath=in_path, word_vocab=word_vocab,
                                   label_vocab=label_vocab,
                                   isShuffle=False, isLoop=True,
                                   max_sent_length=FLAGS.max_sent_length, is_inpath_input_text= is_inpath_input_text)
    return testDataStream


def decode (model_prefix, label_vocab, valid_graph, sess, testDataStream, out_path = None):
    # print('Loading configurations.')
    # FLAGS = load_namespace(model_prefix + ".config.json")
    # print(FLAGS)
    # testDataStream = HanDataStream(inpath=in_path, word_vocab=word_vocab,
    #                                label_vocab=label_vocab,
    #                                isShuffle=False, isLoop=True,
    #                                max_sent_length=FLAGS.max_sent_length)
    if out_path != None:  # decode and write samples in file
        accuracy, confusion_matrix = evaluate(testDataStream, valid_graph, sess,
                                              outpath=out_path, label_vocab=label_vocab)
        print("Accuracy for test set is %.2f" % accuracy)
    else:  # in case of flask, which we only have single instance and want to decect its topic.
        classes = evaluate(testDataStream, valid_graph, sess,
                           label_vocab=label_vocab, mode='flask')
        return classes

# def decode (model_prefix, in_path, label_vocab, word_vocab, out_path = None):
#     # load the configuration file
#     print('Loading configurations.')
#     FLAGS = app.HanClassifier.src.namespace_utils.load_namespace(model_prefix + ".config.json")
#     print(FLAGS)
#
#     num_classes = label_vocab.size()
#
#     testDataStream = HanDataStream(inpath=in_path, word_vocab=word_vocab,
#                                    label_vocab=label_vocab,
#                                    isShuffle=False, isLoop=True,
#                                    max_sent_length=FLAGS.max_sent_length)
#
#     print('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))
#
#     best_path = model_prefix + ".best.model"
#     print('Decoding on the test set:')
#     with tf.Graph().as_default():
#         initializer = tf.contrib.layers.xavier_initializer()
#         with tf.variable_scope("Model", reuse=False, initializer=initializer):
#             valid_graph = HanModelGraph(num_classes=num_classes, word_vocab=word_vocab,
#                                         dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate,
#                                         optimize_type=FLAGS.optimize_type,
#                                         lambda_l2=FLAGS.lambda_l2,
#                                         context_lstm_dim=FLAGS.context_lstm_dim,
#                                         is_training=False, batch_size=1)
#         vars_ = {}
#         for var in tf.all_variables():
#             if "word_embedding" in var.name: continue
#             if not var.name.startswith("Model"): continue
#             vars_[var.name.split(":")[0]] = var
#         saver = tf.train.Saver(vars_)
#
#         sess = tf.Session()
#         sess.run(tf.global_variables_initializer())
#         saver.restore(sess, best_path)
#
#         # print(HanTrainer.evaluate(testDataStream, valid_graph, sess,
#         #                           outpath=out_path, label_vocab=label_vocab,
#         #                           mode='predict'))
#
#         start_time = time.time ()
#         if out_path != None: #decode and write samples in file
#             accuracy, confusion_matrix = evaluate(testDataStream, valid_graph, sess,
#                                                              outpath=out_path, label_vocab=label_vocab)
#             print("Accuracy for test set is %.2f" % accuracy)
#         else: # in case of flask, which we only have single instance and want to decect its topic.
#             classes = evaluate(testDataStream, valid_graph, sess,
#                                                          label_vocab=label_vocab, mode = 'flask')
#             return classes , time.time () - start_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='run6-1', help='Prefix to the models.')
    parser.add_argument('--model_dir', type=str,default = 'app/HanClassifier/models',help='Directory to save model files.')
    parser.add_argument('--in_path', type=str, default='app/HanClassifier/data/news/test10-6.txt', help='the path to the test file.')
    parser.add_argument('--out_path', type=str, default= 'app/HanClassifier/result/test', help='The path to the output file.')
    parser.add_argument('--word_vec_path', type=str, default='app/HanClassifier/data/glove/my_wiki.fa.vec', help='word embedding file for the input file.')

    args, unparsed = parser.parse_known_args()

    model_prefix = args.model_dir + "/Han.{}".format(args.suffix)
    in_path = args.in_path
    out_path = args.out_path + 'S'
    word_vec_path = args.word_vec_path
    word_vocab = load_word_vocab(word_vec_path)
    valid_graph ,sess, label_vocab, FLAGS = \
        load_model(model_prefix,word_vocab)

    testDataStream = load_data_stream(in_path, word_vocab,label_vocab,FLAGS)
    decode(model_prefix, label_vocab, valid_graph, sess, testDataStream, out_path)







        #index = []
        #for i in range(label_vocab.size()):
        #    index.append(label_vocab.getWord (i))
        #columns = index [:]
        #PlotConfusionMatrix.plot(confusion_matrix = confusion_matrix, index = range (num_classes),
        #                         columns= range(num_classes))
