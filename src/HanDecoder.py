from __future__ import print_function
import argparse
import tensorflow as tf
import time


'''
remove . from the first of package for local testing and keep them when you are using flask, i could not find better way :(
'''

from vocab_utils import Vocab
from namespace_utils import load_namespace
from HanTrainer import evaluate
from HanModelGraph import  HanModelGraph
from HanDataStream import HanDataStream

'''load word vocab'''
def load_word_vocab (word_vec_path):
    word_vocab= Vocab(word_vec_path, fileformat='txt3')
    return word_vocab

'''
this function, load model graph based on it model_prefix (model name that we stored)
and return its valid graph, its session, label vocab and FLAGS (config of graph, like hyperparams, etc)
'''
def load_model (model_prefix, word_vocab):
    FLAGS = load_namespace(model_prefix + ".config.json")
    label_vocab = Vocab(model_prefix + ".label_vocab", fileformat='txt2')
    num_classes = label_vocab.size()
    best_path = model_prefix + ".best.model"
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
        print ("ValidGraph Build")
        for var in tf.global_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        config = tf.ConfigProto(intra_op_parallelism_threads=0,
                                inter_op_parallelism_threads=0,
                                allow_soft_placement=True)

        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, best_path)
        return valid_graph, sess, label_vocab, FLAGS
# def load_FLAGS (model_prefix):
#     print('Loading configurations.')
#     FLAGS = load_namespace(model_prefix + ".config.json")
#     print(FLAGS)


'''
load data_stream stored in in_path.
if is_inpath_input_text == True : inpath is the content of a file that must be read, it used when we pass text string not file for loading.
'''
def load_data_stream (in_path, word_vocab, label_vocab, FLAGS, is_inpath_input_text = False):
    testDataStream = HanDataStream(inpath=in_path, word_vocab=word_vocab,
                                   label_vocab=label_vocab,
                                   isShuffle=False, isLoop=True,
                                   max_sent_length=FLAGS.max_sent_length, is_inpath_input_text= is_inpath_input_text)
    return testDataStream

'''
decode the testDataStream based on given valid_graph and its associated session.
if is_flask == True: we dont have any ground truth labels and just wanna predict label for class and return it as predicted class name
    (it is used in flask(deployment))
else: we have ground truth label with testDataStream and we want to evaluate the accurcy of our model and in case of outpath!=None, returen 
    wrong predicted class with their content into outpath file.
if get_class_prob == True: it is used when we want to return top 5 predicted class(based on their probability) with their probability,
    it is only used when is_flask = True
'''
def decode (model_prefix, label_vocab, valid_graph, sess, testDataStream, out_path = None, get_class_prob = False, is_flask = True):
    if is_flask == False:  # decode and evaluate on given test set
        accuracy = evaluate(testDataStream, valid_graph, sess,
                                              outpath=out_path, label_vocab=label_vocab,mode=None, get_class_probs=get_class_prob)
        if out_path != None:
            accuracy, confusion_matrix = accuracy
        print("Accuracy for test set is %.2f" % accuracy)
    else:  # in case of flask, which we only want to detect the topic
        classes = evaluate(testDataStream, valid_graph, sess, outpath= out_path,
                           label_vocab=label_vocab, mode='flask',get_class_probs=get_class_prob)
        return classes # reutrns 2 list, first one is predicted class labels and second one is list of probability,
                        # if get_class_prob = False: first list has only one member and the second one is empty

#def evaluate(dataStream, valid_graph, sess, outpath=None, label_vocab=None, mode = None, get_class_probs = False):


'''
used only for testing the model speed or its accuracy
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='run6-1', help='Prefix to the models.')
    parser.add_argument('--model_dir', type=str,default = '../models',help='Directory to save model files.')
    parser.add_argument('--in_path', type=str, default='../data/news/test1000-6.txt', help='the path to the test file.')
    parser.add_argument('--out_path', type=str, default= '../result/test', help='The path to the output file.')
    parser.add_argument('--word_vec_path', type=str, default='../data/glove/my_wiki.fa.vec', help='word embedding file for the input file.')

    args, unparsed = parser.parse_known_args()

    model_prefix = args.model_dir + "/Han.{}".format(args.suffix)
    in_path = args.in_path
    '''always we add S character to the output file, output file, containes strings that we could not classify them correctly.'''
    out_path = args.out_path + 'S'
    word_vec_path = args.word_vec_path
    word_vocab = load_word_vocab(word_vec_path)
    print("word_vocab_loaded")
    ''' load the model based on model_prefix'''
    valid_graph ,sess, label_vocab, FLAGS = \
        load_model(model_prefix,word_vocab)
    print ("model loaded")
    '''load data stream:'''
    first_time = time.time()
    testDataStream = load_data_stream(in_path, word_vocab,label_vocab,FLAGS)
    print ("data_stream build")
    '''decode data stream on graph based on loaded model'''
    decode(model_prefix, label_vocab, valid_graph, sess, testDataStream, out_path = None, get_class_prob= False, is_flask=False)
    duration = time.time() - first_time
    print ("duration of decoding: {}".format(duration))