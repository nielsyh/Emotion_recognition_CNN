import tensorflow as tf

def conv2d(x, W, b, strides = 1):
    x =  tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x

#default 2x2 maxpooling
def maxpool2d(x, k = 2):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
