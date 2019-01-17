import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from data_handler import Data, plot_acc, plot_tt_acc

data = Data(41904)
train_X = data.train_x #mnist.train.images
test_X = data.test_x #mnist.test.images
train_y = data.train_y #mnist.train.labels
test_y = data.test_y #mnist.test.labels

n_classes = 11
batch_size = 128



x = tf.placeholder('float', [None,1,200,200])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    weights = {
        # 3 x 3 convolution, 1 input image, 32 outputs
        'W_conv1': tf.Variable(tf.random_normal([3, 3, 1, 32])),
        'W_conv2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
        'W_conv3': tf.Variable(tf.random_normal([3, 3, 64, 128])),

        #fully connected after 3xmaxpooling 200 is 25. with input size 128.
        'W_fc': tf.Variable(tf.random_normal([25 * 25 * 128, 1024])),
        # 1024 inputs, 11 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_conv3': tf.Variable(tf.random_normal([128])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    x = tf.reshape(x, shape=[-1, 200, 200, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']+ biases['b_conv3']))
    conv3 = maxpool2d(conv3)

    fc = tf.reshape(conv3, [-1, 25 * 25 * 128])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 2
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for b in range(int(len(train_y) / batch_size)):
                epoch_x = train_X[b * batch_size:min((b + 1) * batch_size, len(train_X))]
                epoch_y = train_y[b * batch_size:min((b + 1) * batch_size, len(train_y))]

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_X, y: test_y}))


train_neural_network(x)