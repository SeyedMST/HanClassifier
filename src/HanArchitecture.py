import tensorflow as tf
from tensorflow.python.ops import rnn


def cal_wxb(in_val, scope, output_dim, input_dim, activation = 'relu'):
    #in_val : [sent_cnt, sent_len, dim]
    input_shape = tf.shape(in_val)
    sent_cnt = input_shape[0]
    sent_len = input_shape[1]
    in_val = tf.reshape(in_val, [sent_cnt*sent_len, input_dim])
    with tf.variable_scope(scope):
        w = tf.get_variable('sim_w', [input_dim, output_dim], dtype=tf.float32)
        b = tf.get_variable('sim_b', [output_dim], dtype = tf.float32)
        if activation == 'relu':
            outputs = tf.nn.relu(tf.nn.xw_plus_b(in_val, w, b))
        elif activation == 'tanh':
            outputs = tf.nn.tanh(tf.nn.xw_plus_b(in_val, w, b))
        else: #None
            outputs = tf.nn.xw_plus_b(in_val, w, b)
    outputs = tf.reshape(outputs, [sent_cnt, sent_len, output_dim])
    return outputs # [sent_cnt, sent_len, output_dim]


def self_attention(text_context_representation_fw, text_context_representation_bw,mask,input_dim):
    text_context_representation_bw = tf.multiply(text_context_representation_bw,
                                                     tf.expand_dims(mask, -1))
    text_context_representation_fw = tf.multiply(text_context_representation_fw,
                                                     tf.expand_dims(mask, -1))
    text_rep = tf.concat([text_context_representation_fw,text_context_representation_bw], 2)
    shrinking_factor = 2
    res = cal_wxb(text_rep, scope='context_self_att_1',
                            output_dim=input_dim/shrinking_factor, input_dim=input_dim,activation='tanh') #[sent_cnt, sent_len, dim/2]
    res = cal_wxb(res, scope='context_self_att_2',
                 output_dim=1, input_dim=input_dim/shrinking_factor, activation='None')#[sent_cnt, sent_len, 1]
    agg_shape = tf.shape(res)
    sent_cnt = agg_shape[0]
    sent_len = agg_shape[1]
    res = tf.reshape(res, [sent_cnt, sent_len])  # [sent_cnt, sent_len]
    res = tf.nn.softmax(res)
    res = tf.expand_dims(res, axis=-1)  # [sent_cnt, sent_len, 1]
    res = tf.multiply(res, text_rep)  # [sent_cnt, sent_len, dim]
    res = tf.reduce_mean(res, axis=1)  # [sent_cnt,dim]
    return res



#return re
def HanArc (in_text_repres, sents_length,  mask, input_dim,
                        context_lstm_dim,is_training,dropout_rate):
    with tf.variable_scope('context_represent'):

        context_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(context_lstm_dim)
        context_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(context_lstm_dim)
        if is_training ==True:
            context_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_fw,
                                                                 output_keep_prob=(1 - dropout_rate))
            context_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_bw,
                                                                 output_keep_prob=(1 - dropout_rate))
        context_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([context_lstm_cell_fw])
        context_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([context_lstm_cell_bw])

        # text representation
        (text_context_representation_fw, text_context_representation_bw), _ = rnn.bidirectional_dynamic_rnn(
            context_lstm_cell_fw, context_lstm_cell_bw, in_text_repres, dtype=tf.float32,
            sequence_length=sents_length)  # [sent_cnt, sent_len, context_lstm_dim]
        # in_text_repres = tf.concat([text_context_representation_fw,
        #                             text_context_representation_bw], 2) # [sent_cnt,sent_len,2*context_lstm_dim]
        # in_text_repres = self_attention(text_context_representation_fw, text_context_representation_bw, mask,
        #                context_lstm_dim * 2)

        fw_rep = text_context_representation_fw[:, -1, :] #[sent_cnt, dim]
        bw_rep = text_context_representation_bw[:, 0, :] #[sent_cnt, dim]
        final_sent_repres = tf.concat([fw_rep, bw_rep], 1) #[sent_cnt, dim]
        #average of sentence representations is final rep:
        final_text_repres = tf.reduce_mean(final_sent_repres, axis=0) #[dim]
        final_text_repres_dim = 2*context_lstm_dim
        return final_text_repres, final_text_repres_dim # dim

