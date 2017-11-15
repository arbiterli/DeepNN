from matplotlib import pyplot as plt

import tensorflow as tf
import numpy as np


mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

data = train_data[1]
data = data.reshape([28,28])

print(data)
plt.gray()
plt.imshow(data, interpolation='nearest')
plt.show()