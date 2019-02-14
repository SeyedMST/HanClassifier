import tensorflow as tf
#import my_rnn
import numpy as np
from tensorflow.python.ops.rnn import dynamic_rnn
import HanArchitecture


eps = 1e-8



#this function gives a tuple of place holders, the length of tuple is batch size
def get_place_holder (batch_size, type, shape):
    ans = ()
    for _ in range(batch_size):
        ans += (tf.placehbolder(type, shape=shape),)
    return ans


class HanModelGraph(object):
    def __init__(self, num_classes, word_vocab,
                 dropout_rate, learning_rate, optimize_type, lambda_l2,
                 context_lstm_dim, is_training, batch_size):

        self.sents_length = get_place_holder (batch_size, tf.int32, [None]) #[batch_size, sent_cnt]
        self.truth = tf.placeholder(tf.int32, [None]) # [batch_size]

        self.in_text_words = get_place_holder (batch_size, tf.int32, [None, None])#[batch_size, sent_cnt, max_sent_len]

        with tf.device('/cpu:0'):
            self.word_embedding = tf.get_variable("word_embedding", trainable=False,
                                              initializer=tf.constant(word_vocab.word_vecs), dtype=tf.float32)

        with tf.variable_scope('HanGraph'):
            for i in range (batch_size):
                in_text_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_text_words[i]) # [sent_cnt, max_sent_len ,word_dim]
                input_dim = word_vocab.word_dim
                if is_training:
                   in_text_repres = tf.nn.dropout(in_text_repres, (1 - dropout_rate))
                else:
                   in_text_repres = tf.multiply(in_text_repres, (1 - dropout_rate))

                input_shape = tf.shape(self.in_text_words[i])
                max_sent_len = input_shape[1]

                mask = tf.sequence_mask(self.sents_length[i], max_sent_len, dtype=tf.float32) # [sent_cnt, max_sent_len]

                text_rep, text_rep_dim = HanArchitecture.HanArc(in_text_repres, self.sents_length, mask, input_dim,
                context_lstm_dim, is_training, dropout_rate)









