import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from network import *
from data_handler import Data, plot_acc, plot_tt_acc

#help websites
#source https://pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-machine-learning-tutorial/
#source https://www.datacamp.com/community/tutorials/cnn-tensorflow-python

#init data
data = Data(41904)
train_X = data.train_x #mnist.train.images
test_X = data.test_x #mnist.test.images
train_y = data.train_y #mnist.train.labels
test_y = data.test_y #mnist.test.labels

#placeholder for data = amount(differs), channels, height, width
x = tf.placeholder('float', [None,1,200,200])
y = tf.placeholder('float')

#important settings
keep_rate = 0.25 #dropout rate
keep_prob = tf.placeholder(tf.float32)
epochs = 10 # howmany dataset is pushed trough network... This is the slow part
n_classes = 11
batch_size = 128 # howmany samples at once trough network.


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

    # Reshape input to a 4D tensor
    x = tf.reshape(x, shape=[-1, 200, 200, 1])

    # Convolution Layer, using our function
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1'], biases['b_conv1']))
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2'], biases['b_conv2']))
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2)

    conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3'], biases['b_conv3']))
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3)

    fc = tf.reshape(conv3, [-1, 25 * 25 * 128])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']
    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels=y)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer().minimize(cost) #default learning rate? #todo what parameters/optimizer
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_loss = []
        test_loss = []
        train_acc = []
        test_acc = []

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        for epoch in range(epochs):
            epoch_loss = 0

            for b in range(int(data.num_train()/ batch_size)):


                epoch_x = train_X[b * batch_size:min((b + 1) * batch_size, len(train_X))]
                epoch_y = train_y[b * batch_size:min((b + 1) * batch_size, len(train_y))]

                opt = sess.run(optimizer, feed_dict={x: epoch_x, y: epoch_y})
                loss, acc = sess.run([cost, accuracy], feed_dict={x: epoch_x,
                                                                  y: epoch_y})

            # Calculate accuracy for all 10000 mnist test images
            test_accu, valid_loss = sess.run([accuracy, cost], feed_dict={x: epoch_x, y: epoch_y})
            train_loss.append(loss)
            test_loss.append(valid_loss)
            train_acc.append(acc)
            test_acc.append(test_accu)

            print("epoch " + str(epoch) + ", Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            print("Test accuracy: " + str(test_accu))

        print('Accuracy:', accuracy.eval({x: data.test_x, y: data.test_y}))
        plot_acc(train_loss, test_loss)
        plot_tt_acc(train_loss, train_acc, test_acc)


train_neural_network(x)