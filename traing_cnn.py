import tensorflow as tf
import numpy as np
from CNN_model.CNN import CNN


def get_batch_data(index, _train_data, _train_labels, _batch_size):
    start = index * _batch_size
    end = start + _batch_size
    return _train_data[start:end][:], _train_labels[start:end][:]


def main(_):
    # Load data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    cnn = CNN()
    batch_size = 100
    batch_num = int(train_data.shape[0] / batch_size)
    total_times = 40
    sv = tf.train.Supervisor(graph=cnn.graph,
                             logdir='./log',
                             save_model_secs=0)

    with sv.managed_session() as sess:
        for time in range(total_times):
            for i in range(batch_num):
                x_batch, y_batch = get_batch_data(i, train_data, train_labels, batch_size)
                train_feed_dict = {cnn.images: x_batch,
                                   cnn.labels: y_batch}
                _, loss, training_step = sess.run([cnn.train_op, cnn.loss, cnn.global_step], train_feed_dict)
                if training_step % 50 == 0:
                    print(loss, training_step)

        # evaluate on test dataset
        eval_feed_dict = {cnn.images: eval_data}
        results = sess.run(cnn.output, eval_feed_dict)
        for index in range(results.shape[0]):
            result = np.argmax(results[index][:])
            print(result, eval_labels[index])


if __name__ == '__main__':
    tf.app.run()
