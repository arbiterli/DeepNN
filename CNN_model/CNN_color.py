import tensorflow as tf
import matplotlib.image as mpimg


image = '../models/koala.jpg'
image_data = tf.read_file(image)
decode_data = tf.image.decode_jpeg(image_data, channels=3)
with tf.Session().as_default():
    print(decode_data.eval())
