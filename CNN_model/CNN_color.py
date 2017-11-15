import tensorflow as tf
import matplotlib.image as mpimg


image = '../models/koala.jpg'
# image_data = tf.gfile.FastGFile(image, 'rb').read()
# decode_data = tf.image.decode_jpeg(image_data, channels=3)
# print(decode_data)

image_data = mpimg.imread(image)
print(image_data.shape)