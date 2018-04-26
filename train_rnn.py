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
    normalized_aim_data = np.zeros((aim_data.shape[0], aim_data.shape[1], 12))
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
    normalized_aim_data[:, :, 11] = dis
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


def get_supervised_batch_data(index, data, label, batch_size):
    start_pos = index * batch_size
    return data[start_pos:start_pos+batch_size][:], label[start_pos:start_pos+batch_size][:]


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


def evaluate(sess, rnn_model, eval_data, eval_label, data_flag):
    eval_X, eval_Y = get_supervised_batch_data(0, eval_data, eval_label, batch_size=eval_data.shape[0])
    eval_feed_dict = {rnn_model.X: eval_X}
    results = sess.run(rnn_model.output, eval_feed_dict)
    corr_cheat = 0
    total_cheat = 0
    corr_normal = 0
    total_normal = 0
    for i in range(results.shape[0]):
        if eval_Y[i][0] == 1:
            total_cheat += 1
            if results[i][0] > results[i][1]:
                corr_cheat += 1
        else:
            total_normal += 1
            if results[i][0] < results[i][1]:
                corr_normal += 1
                # print(results[i], eval_Y[i])
    print(data_flag, 'cheat accuracy:', corr_cheat / total_cheat)
    print(data_flag, 'normal accuracy:', corr_normal / total_normal)
    return corr_cheat / total_cheat, corr_normal / total_normal


def main(_):
    cheat_train_data, cheat_test_data = split_data('aim_data/cheat_aim_data.bin')
    normal_train_data, normal_test_data = split_data('aim_data/normal_aim_data.bin')
    train_data_size = cheat_train_data.shape[0] + normal_train_data.shape[0]
    train_data = np.zeros((train_data_size, cheat_train_data.shape[1], cheat_train_data.shape[2]))
    # binary classification, (1, 0)->cheat, (0, 1)->normal
    train_label = np.zeros((train_data_size, 2))
    train_data[:cheat_train_data.shape[0]][:] = cheat_train_data
    train_data[cheat_train_data.shape[0]:][:]= normal_train_data
    train_label[:cheat_train_data.shape[0], 0] = np.ones(cheat_train_data.shape[0])
    train_label[cheat_train_data.shape[0]:, 1] = np.ones(normal_train_data.shape[0])

    test_data_size = cheat_test_data.shape[0] + normal_test_data.shape[0]
    test_data = np.zeros((test_data_size, cheat_test_data.shape[1], cheat_test_data.shape[2]))
    test_label = np.zeros((test_data_size, 2))
    test_data[:cheat_test_data.shape[0]][:] = cheat_test_data
    test_data[cheat_test_data.shape[0]:][:] = normal_test_data
    test_label[:cheat_test_data.shape[0], 0] = np.ones(cheat_test_data.shape[0])
    test_label[cheat_test_data.shape[0]:, 1] = np.ones(normal_test_data.shape[0])

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    rnn = RNN(time_step=train_data.shape[1], vector_size=train_data.shape[2], label_size=train_label.shape[1])
    batch_size = train_data.shape[0]
    total_times = 10000
    batch_num = int(train_data_size / batch_size)

    sv = tf.train.Supervisor(graph=rnn.graph,
                             logdir='./log',
                             save_model_secs=0)

    f = open('results.txt', 'w')

    # config=tf.ConfigProto(log_device_placement=True)
    draw_x = list()
    train_cheat_accs = list()
    train_normal_accs = list()
    test_cheat_accs = list()
    test_normal_accs = list()
    with sv.managed_session() as sess:
        sess.run(rnn.init_op)
        # logdir = "tensorboard/"
        # writer = tf.summary.FileWriter(logdir, sess.graph)
        for time in range(total_times):
            print('time: ' + str(time))
            for i in range(batch_num):
                x_batch, y_batch = get_supervised_batch_data(i, train_data, train_label, batch_size)
                train_feed_dict = {rnn.X: x_batch,
                                   rnn.Y: y_batch}
                _, loss, o, training_step = sess.run([rnn.train_op, rnn.loss, rnn.output,
                                                           rnn.global_step], train_feed_dict)
                if training_step % 10 == 0:
                    print(loss, training_step)
                    # evaluate on training set
                    train_cheat_acc, train_normal_acc = evaluate(sess, rnn, train_data, train_label, data_flag='train')
                    # evaluate on test dataset
                    test_cheat_acc, test_normal_acc = evaluate(sess, rnn, test_data, test_label, data_flag='test')
                    draw_x.append(training_step)
                    train_cheat_accs.append(train_cheat_acc)
                    train_normal_accs.append(train_normal_acc)
                    test_cheat_accs.append(test_cheat_acc)
                    test_normal_accs.append(test_normal_acc)
                    print('==============================')
                    # writer.add_summary(summary, training_step)
            if loss < 0.01:
                break
        # writer.close()

    print(len(train_cheat_accs))
    # draw plot for accurary
    import plotly.graph_objs as go
    import plotly as py

    traces = list()
    trace_train_cheat = go.Scatter(x=draw_x, y=train_cheat_accs, mode='lines', name='train cheat acc')
    trace_train_normal = go.Scatter(x=draw_x, y=train_normal_accs, mode='lines', name='train normal acc')
    trace_test_cheat = go.Scatter(x=draw_x, y=test_cheat_accs, mode='lines', name='test cheat acc')
    trace_test_normal = go.Scatter(x=draw_x, y=test_normal_accs, mode='lines', name='test normal acc')
    traces.append(trace_train_cheat)
    traces.append(trace_train_normal)
    traces.append(trace_test_cheat)
    traces.append(trace_test_normal)
    fig = dict(data=traces)
    py.offline.plot(fig, filename='result.html', auto_open=False)


if __name__ == '__main__':
    tf.app.run()
