import tensorflow as tf


class RNN():
    def __init__(self, time_step, vector_size, label_size, batch_size=None):
        self.hidden_unit_size = 32
        self.num_layers = 2
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape=[batch_size, time_step, vector_size])
            self.Y = tf.placeholder(tf.float32, shape=[batch_size, label_size])
            self.build_graph(label_size)
            self.init_op = tf.global_variables_initializer()
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.output, self.Y)))
            self.optimize()
            # tf.summary.scalar('loss', self.loss)

    def build_graph(self, label_size):
        with tf.variable_scope('LSTM'):
            cells = list()
            for i in range(self.num_layers):
                cells.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_unit_size))
                # cells.append(tf.nn.rnn_cell.DropoutWrapper(
                #     tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_unit_size),
                #     output_keep_prob=0.6))
            lstm_cells = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
            self.hidden_output, self.state = tf.nn.dynamic_rnn(cell=lstm_cells, inputs=self.X, dtype=tf.float32)

        weight = tf.Variable(tf.truncated_normal([self.hidden_unit_size, label_size]))
        bias = tf.Variable(tf.zeros([label_size]))
        self.output = tf.matmul(self.hidden_output[:, -1, :], weight) + bias
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.output))


        # with tf.variable_scope('dense'):
        #     # last_hidden_output = tf.gather(self.hidden_output, self.hidden_output.get_shape()[0] - 1)
        #     dense = tf.layers.dense(inputs=self.hidden_output[:, -1, :], units=128, activation=tf.nn.relu)
        #
        # with tf.variable_scope('dropout'):
        #     dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)
        #
        # with tf.variable_scope('output'):
        #     self.output = tf.layers.dense(inputs=dropout, units=label_size)


    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # self.gradient = optimizer.compute_gradients(self.loss)
        # self.train_op = optimizer.apply_gradients(self.gradient, global_step=self.global_step)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)


if __name__ == '__main__':
    import numpy as np
    rnn = RNN(time_step=5, vector_size=10, label_size=2)
    inputs = np.random.randn(2, 5, 10)
    print(inputs)
    feed_dict = {rnn.X: inputs}

    with tf.Session(graph=rnn.graph) as sess:
        sess.run(rnn.init_op)
        result = sess.run(rnn.output, feed_dict=feed_dict)
        print(result)
