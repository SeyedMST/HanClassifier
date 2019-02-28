import tensorflow as tf
'''
remove . from the first of package for local testing and keep them when you are using flask, i could not find better way :(
'''
from HanArchitecture import HanArc


eps = 1e-8
#this function gives a tuple of place holders, the length of tuple is batch size
def get_place_holder (batch_size, type, shape):
    ans = ()
    for _ in range(batch_size):
        ans += (tf.placeholder(type, shape=shape),)
    return ans


class HanModelGraph(object):
    def __init__(self, num_classes, word_vocab,
                 dropout_rate, learning_rate, optimize_type, lambda_l2,
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
            print (num_classes)
            gold_matrix = tf.one_hot(self.truth, num_classes, dtype=tf.float32) #[batch_size, num_classes]
            print (tf.shape(gold_matrix))
            print (tf.shape(self.batch_class_scores))
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                                       (logits = self.batch_class_scores, labels= gold_matrix))

            #optimization:
            if optimize_type == 'adadelta':
                clipper = 50
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
                tvars = tf.trainable_variables()
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
                self.loss = self.loss + lambda_l2 * l2_loss
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
                self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            elif optimize_type == 'sgd':
                self.global_step = tf.Variable(0, name='global_step',
                                               trainable=False)  # Create a variable to track the global step.
                min_lr = 0.000001
                self._lr_rate = tf.maximum(min_lr,
                                           tf.train.exponential_decay(learning_rate, self.global_step, 30000, 0.98))
                self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self._lr_rate).minimize(self.loss)
            elif optimize_type == 'ema':
                tvars = tf.trainable_variables()
                train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
                # Create an ExponentialMovingAverage object
                ema = tf.train.ExponentialMovingAverage(decay=0.9999)
                # Create the shadow variables, and add ops to maintain moving averages # of var0 and var1.
                maintain_averages_op = ema.apply(tvars)
                # Create an op that will update the moving averages after each training
                # step.  This is what we will use in place of the usual training op.
                with tf.control_dependencies([train_op]):
                    self.train_op = tf.group(maintain_averages_op)
            elif optimize_type == 'adam':
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







