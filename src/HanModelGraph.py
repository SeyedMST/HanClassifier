import tensorflow as tf
import numpy as np
'''
remove . from the first of package for local testing and keep them when you are using flask, i could not find better way :(
'''
from HanArchitecture import HanArc


eps = 1e-8


'''
this function, returns feed_dict based on input data_stream and exactly batch_size count.
It is absoulutley possible that len (data_stream) % batch_size != 0, it means that final_batch instance count might be less than batch_size, In this case
we goback to the begining of the data_stream and read data again:
    if is_testing == True:
        we return batch_valid_instances as a parameter that indicate the number of valid instances in the batch (other instances must not include in the result)
    else:
        It's not important, a little data, includes in training more than other, but as we shuffle data, it is not biased to anything :)
'''

def get_feed_dict (data_stream, graph, batch_size, is_testing):
    _truth = []
    _sents_length = []
    _in_text_words = []
    if is_testing == True:
        total_data_stream_instances = data_stream.get_num_instance ()
        total_remaning_instances = total_data_stream_instances - data_stream.get_cur_pointer ()
        batch_valid_instances = min ((batch_size, total_remaning_instances))
    for i in range(batch_size):
        cur_instance, instance_index = data_stream.nextInstance()
        (label, text, label_id, word_idx, sents_length) = cur_instance

        _truth.append(label_id)
        _sents_length.append(sents_length)
        _in_text_words.append(word_idx)

    feed_dict = {
        graph.truth: np.array(_truth),
        graph.sents_length: tuple(_sents_length),
        graph.in_text_words: tuple(_in_text_words),
    }

    if is_testing == True:
        return feed_dict, batch_valid_instances
    else:
        return feed_dict


#this function gives a tuple of place holders, the length of tuple is batch size
def get_place_holder (batch_size, type, shape):
    ans = ()
    for _ in range(batch_size):
        ans += (tf.placeholder(type, shape=shape),)
    return ans


class HanModelGraph(object):
    def __init__(self, num_classes, word_vocab,
                 dropout_rate, learning_rate, lambda_l2,
                 context_lstm_dim, is_training, batch_size):
        print (batch_size)

        self.sents_length = get_place_holder (batch_size, tf.int32, [None]) #[batch_size, sent_cnt]
        self.truth = tf.placeholder(tf.int32, [None]) # [batch_size]
        self.in_text_words = get_place_holder (batch_size, tf.int32, [None, None])#[batch_size, sent_cnt, max_sent_len]

        with tf.device('/cpu:0'):
            self.word_embedding = tf.get_variable("word_embedding", trainable=False,
                                              initializer=tf.constant(word_vocab.word_vecs), dtype=tf.float32)
        with tf.variable_scope('HanGraph'):
            class_scores_list = []
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

                class_scores = HanArc(in_text_repres, self.sents_length[i], mask, input_dim,
                context_lstm_dim, is_training, dropout_rate, num_classes) #[1, num_classes]
                class_scores_list.append(class_scores)
                tf.get_variable_scope().reuse_variables()

        print (len (class_scores_list))
        self.batch_class_scores = tf.concat(class_scores_list, axis=0) # [batch_size, num_classes]
        print (tf.shape (self.batch_class_scores))
        self.prob = tf.nn.softmax(self.batch_class_scores) # [batch_size, num_classes]
        self.predictions = tf.argmax(self.batch_class_scores, 1) #[batch_size]
        correct = tf.nn.in_top_k(self.batch_class_scores, self.truth, 1)
        self.eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32)) #count of correct preds

        if is_training == True: #train and optimeze
            gold_matrix = tf.one_hot(self.truth, num_classes, dtype=tf.float32) #[batch_size, num_classes]
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                                       (logits = self.batch_class_scores, labels= gold_matrix))
            #optimization:
            clipper = 50
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            tvars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

            extra_train_ops = []
            train_ops = [self.train_op] + extra_train_ops
            self.train_op = tf.group(*train_ops)







