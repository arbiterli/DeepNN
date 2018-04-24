import tensorflow as tf
import pickle
import numpy as np
from CNN_model.CNN_color import CNNColor


def read_image(image_path):
    image_data = tf.read_file(image_path)
    return tf.image.decode_jpeg(image_data, channels=3)


def get_batch_data(index, _train_data, _train_labels, _batch_size):
    start = index * _batch_size
    end = start + _batch_size
    return _train_data[start:end][:], _train_labels[start:end][:]


def get_data():
    train_data = np.zeros([50000, 32, 32, 3])
    train_labels = np.zeros([50000])
    pickle_file_pre = 'data/cifar-10-batches-py/data_batch_'
    for img_id in range(5):
        pickle_file = pickle_file_pre + str(img_id+1)
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        arr = np.array(data[b'data'])
        train_labels[img_id*10000:(img_id+1)*10000] = data[b'labels']
        train_data[img_id*10000:(img_id+1)*10000][:] = arr.reshape([10000, 3, 32, 32]).transpose(0, 2, 3, 1)

    with open('data/cifar-10-batches-py/test_batch', 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')
    test_arr = np.array(test_data[b'data'])
    test_labels = np.array(test_data[b'labels'])
    test_image = test_arr.reshape([10000, 3, 32, 32]).transpose(0, 2, 3, 1)
    return train_data, train_labels, test_image, test_labels


def main(_):
    train_image, train_labels, test_image, test_labels = get_data()
    print(train_image.shape, train_labels.shape, test_image.shape, test_labels.shape)
    print(train_labels)
    batch_data = get_batch_data(0, train_image, train_labels, 100)
    print(batch_data[0].shape)

    cnn_color = CNNColor(32, 32)
    batch_size = 100
    batch_num = int(train_image.shape[0] / batch_size)
    print(batch_num)
    total_times = 30
    sv = tf.train.Supervisor(graph=cnn_color.graph,
                             logdir='./log',
                             save_model_secs=0)

    with sv.managed_session() as sess:
        try:
            for time in range(total_times):
                for i in range(batch_num):
                    x_batch, y_batch = get_batch_data(i, train_image, train_labels, batch_size)
                    train_feed_dict = {cnn_color.X: x_batch,
                                       cnn_color.labels: y_batch}
                    _, loss, training_step = sess.run([cnn_color.train_op, cnn_color.loss, cnn_color.global_step], train_feed_dict)
                    if training_step % 50 == 0:
                        print(loss, training_step)
        except KeyboardInterrupt as e:
            print('done training by {}'.format(str(e)))

        # test data evaluation
        c = 0
        e = 0
        for i in range(int(test_image.shape[0] / batch_size)):
            test_batch_X, test_batch_Y = get_batch_data(i, test_image, test_labels, batch_size)
            eval_feed_dict = {cnn_color.X: test_batch_X}
            results = sess.run(cnn_color.output, eval_feed_dict)

            for index in range(results.shape[0]):
                result = np.argmax(results[index][:])
                if result == test_batch_Y[index]:
                    c += 1
                else:
                    e += 1
                print(result, test_batch_Y[index])
        print(c, e)


if __name__ == '__main__':
    tf.app.run()
