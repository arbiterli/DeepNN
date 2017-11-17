import tensorflow as tf


class RNN():
    def __init__(self, max_time=5, embedding_size=10, batch_size=None):
        self.hidden_unit_size = 32
        self.num_layers = 2
        self.graph = tf.Graph()

        with self.graph.as_default():

            self.X = tf.placeholder(tf.float32, shape=[batch_size, max_time, embedding_size])
            self.Y = tf.placeholder(tf.float32, shape=[batch_size, max_time, embedding_size])
            self.build_graph()
            self.init_op = tf.global_variables_initializer()

    def build_graph(self):
        with tf.variable_scope('LSTM'):
            self.cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_unit_size)
            # self.cell_layer = tf.nn.rnn_cell.MultiRNNCell([self.cell]*self.num_layers)
            self.hidden_output, self.state = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.X, dtype=tf.float32)

        with tf.variable_scope('dense'):
            self.output = tf.layers.dense(inputs=self.hidden_output, units=10)


if __name__ == '__main__':
    import numpy as np
    rnn = RNN()
    inputs = np.random.randn(2, 5, 10)
    print(inputs)
    feed_dict = {rnn.X: inputs}

    with tf.Session(graph=rnn.graph) as sess:
        sess.run(rnn.init_op)
        result = sess.run(rnn.hidden_output, feed_dict=feed_dict)
        print(result.shape)
