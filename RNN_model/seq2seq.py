import tensorflow as tf
import tensorflow.contrib.seq2seq as s2s


class S2SNN():
    def __init__(self, max_time=5, embedding_size=10, batch_size=None):
        self.hidden_unit_size = 32
        self.num_layers = 2
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape=[batch_size, max_time, embedding_size])
            self.Y = tf.placeholder(tf.float32, shape=[batch_size, max_time, embedding_size])
            self.sequence_length = tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(self.X), 2)), 1)
            self.build_encoder()
            self.init_op = tf.global_variables_initializer()
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.Y, logits=self.outputs)
            self.optimize()

    def build_encoder(self):
        with tf.variable_scope('LSTM'):
            cells = list()
            for i in range(self.num_layers):
                # use this when need dropout
                cells.append(tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_unit_size),
                    output_keep_prob=0.5))

            self.cell_layers = tf.nn.rnn_cell.MultiRNNCell(cells)
            self.hidden_output, self.hidden_state = tf.nn.dynamic_rnn(cell=self.cell_layers, inputs=self.X, dtype=tf.float32)

    # [batch_size, max_time, shape]
    def build_decoder(self):
        self.decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_unit_size)
        self.helper = s2s.TrainingHelper(inputs=self.Y, sequence_length=self.sequence_length)
        # pass top layer state of encoder
        self.decoder = s2s.BasicDecoder(cell=self.decoder_cell, helper=self.helper, initial_state=self.hidden_state[1])
        self.decoder_outputs, self.decoder_states, self.decoder_sequence_length = s2s.dynamic_decode(decoder=self.decoder)

    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)


if __name__ == '__main__':
    import numpy as np
    rnn = S2SNN()
    inputs = np.random.randn(2, 5, 10)
    print(inputs)
    feed_dict = {rnn.X: inputs}

    with tf.Session(graph=rnn.graph) as sess:
        sess.run(rnn.init_op)
        result = sess.run(rnn.hidden_output, feed_dict=feed_dict)
        print(result.shape)
