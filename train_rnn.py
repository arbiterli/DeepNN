import numpy as np
import tensorflow as tf
from RNN_model.RNN import RNN


def load_aim_data_from_buffer(file_path):
    with open(file_path, 'rb') as f:
        data = np.frombuffer(f.read()).reshape(-1, 40, 10)
    return data


def normalize(aim_data):
    # aim_data: (0,1,2) killer position (x, y, z), (3,4,5) victim position,
    # (6,7) killer viewangle (x, y), (8,9) victim viewangle (x, y)
    normalized_aim_data = np.zeros((aim_data.shape[0], aim_data.shape[1], 11))
    normalized_aim_data[:, :, 0] = aim_data[:, :, 3] - aim_data[:, :, 0]
    normalized_aim_data[:, :, 1] = aim_data[:, :, 4] - aim_data[:, :, 1]
    normalized_aim_data[:, :, 2] = aim_data[:, :, 5] - aim_data[:, :, 2]
    dis = np.sqrt(normalized_aim_data[:, :, 0]*normalized_aim_data[:, :, 0] +\
          normalized_aim_data[:, :, 1] * normalized_aim_data[:, :, 1] +\
          normalized_aim_data[:, :, 2] * normalized_aim_data[:, :, 2])
    normalized_aim_data[:, :, 0] /= dis
    normalized_aim_data[:, :, 1] /= dis
    normalized_aim_data[:, :, 2] /= dis
    normalized_aim_data[:, :, 3] = np.sin(aim_data[:, :, 6])
    normalized_aim_data[:, :, 4] = np.cos(aim_data[:, :, 6])
    normalized_aim_data[:, :, 5] = np.sin(aim_data[:, :, 7])
    normalized_aim_data[:, :, 6] = np.cos(aim_data[:, :, 7])
    normalized_aim_data[:, :, 7] = np.sin(aim_data[:, :, 8])
    normalized_aim_data[:, :, 8] = np.cos(aim_data[:, :, 8])
    normalized_aim_data[:, :, 9] = np.sin(aim_data[:, :, 9])
    normalized_aim_data[:, :, 10] = np.cos(aim_data[:, :, 9])
    return normalized_aim_data


# _num_damage_event * batches_per_damage = batch size
def get_batch_data(index, _train_data, _num_damage_event=3, _input_seq_length=10):
    batches_per_damage = _train_data.shape[1] - _input_seq_length
    raw_input_seq = np.zeros([_num_damage_event * batches_per_damage, _input_seq_length, 11])
    labels = np.zeros([_num_damage_event * batches_per_damage, 4])
    for i in range(_num_damage_event):
        for j in range(batches_per_damage):
            raw_input_seq[i*batches_per_damage + j][:] =\
                _train_data[index*_num_damage_event+i][j:j+_input_seq_length, :]

            # only use aim direction as labels
            labels[i*batches_per_damage + j][:] = _train_data[index*_num_damage_event+i][j+_input_seq_length, 3:7]
    return raw_input_seq, labels


def get_supervised_batch_data(index, train_data, train_label, batch_size):
    return train_data[index:index+batch_size][:], train_label[index:index+batch_size][:]


def flatten(seq):
  for el in seq:
    if isinstance(el, list):
      yield from flatten(el)
    else:
      yield


def split_data(file_path):
    aim_data = load_aim_data_from_buffer(file_path)
    normalized_aim_data = normalize(aim_data)
    data_size = normalized_aim_data.shape[0]
    train_data_size = int(data_size * 0.8)
    train_data = normalized_aim_data[:train_data_size][:]
    test_data = normalized_aim_data[train_data_size:][:]
    print(normalized_aim_data.shape)
    return train_data, test_data


def main(_):
    cheat_train_data, cheat_test_data = split_data('aim_data/cheat_aim_data.bin')
    normal_train_data, normal_test_data = split_data('aim_data/cheat_aim_data.bin')
    train_data_size = cheat_train_data.shape[0] + normal_train_data.shape[0]
    train_data = np.zeros((train_data_size, cheat_train_data.shape[1], cheat_train_data.shape[2]))
    # binary classification, (1, 0)->cheat, (0, 1)->normal
    train_label = np.zeros((train_data_size, 2))
    train_data[:cheat_train_data.shape[0]][:] = cheat_train_data
    train_data[cheat_train_data.shape[0]:][:]= normal_train_data
    train_label[:cheat_train_data.shape[0], 0] = np.ones(cheat_train_data.shape[0])
    train_label[cheat_train_data.shape[0]:, 1] = np.ones(normal_train_data.shape[0])
    print(train_data.shape, train_label.shape)
    print(train_data)
    print('===========================')
    print(train_label)
    rnn = RNN(time_step=40, vector_size=10, label_size=2)  # n_gram = 10
    batch_size = 100
    total_times = 40
    batch_num = int(train_data_size / batch_size)


    sv = tf.train.Supervisor(graph=rnn.graph,
                             logdir='./log',
                             save_model_secs=0)

    f = open('results.txt', 'w')

    # config=tf.ConfigProto(log_device_placement=True)

    with sv.managed_session() as sess:
        # logdir = "tensorboard/"
        # writer = tf.summary.FileWriter(logdir, sess.graph)
        for time in range(total_times):
            print('time: ' + str(time))
            for i in range(batch_num):
                x_batch, y_batch = get_supervised_batch_data(i, train_data, train_label, batch_num)
                train_feed_dict = {rnn.X: x_batch,
                                   rnn.Y: y_batch}
                _, loss, o, gvs, training_step = sess.run([rnn.train_op, rnn.loss, rnn.output,
                                                           rnn.gradient, rnn.global_step], train_feed_dict)

                if training_step % 100 == 0:
                    print(loss, training_step)
                    print(x_batch[-1, -3:, :])
                    print(gvs[0])
                    print('==============================')
                    # writer.add_summary(summary, training_step)
        # writer.close()

        # evaluate on test dataset
        # eval_X, eval_Y = get_batch_data(0, test_data)
        # eval_feed_dict = {rnn.X: eval_X}
        # print(eval_X.shape, eval_Y.shape)
        # results = sess.run(rnn.output, eval_feed_dict)
        # print(results.shape)
    for index in range(100):
        f.write('X: {}\n'.format(str(eval_X[index])))
        f.write('Y: {}\n'.format(str(eval_Y[index])))
        f.write('predicted_Y: {}\n'.format(str(results[index])))
        f.write('=============================\n\n')
    f.close()

if __name__ == '__main__':
    tf.app.run()