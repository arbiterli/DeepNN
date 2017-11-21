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
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.Y, logits=self.output)
            self.optimize()

    def build_graph(self):
        with tf.variable_scope('LSTM'):
            cells = list()
            for i in range(self.num_layers):
                # tf.nn.rnn_cell.DropoutWrapper: use this when need dropout
                cells.append(tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_unit_size))
            self.cell_layers = tf.nn.rnn_cell.MultiRNNCell(cells)
            self.hidden_output, self.state = tf.nn.dynamic_rnn(cell=self.cell_layers, inputs=self.X, dtype=tf.float32)

        with tf.variable_scope('dense'):
            self.output = tf.layers.dense(inputs=self.hidden_output, units=10)

    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)


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
