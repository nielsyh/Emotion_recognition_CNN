import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from network import *
from data_handler import Data

#source https://pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-machine-learning-tutorial/
#source https://www.datacamp.com/community/tutorials/cnn-tensorflow-python

#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

data = Data(41904)



train_X = data.train_x #mnist.train.images
test_X = data.test_x #mnist.test.images
train_y = data.train_y #mnist.train.labels
test_y = data.test_y #mnist.test.labels


# x = tf.placeholder('float', [None, 784]) 200*200 = 40000
x = tf.placeholder('float', [None, 40000])
y = tf.placeholder('float')

#important settings
keep_rate = 0.5 #dropout rate
keep_prob = tf.placeholder(tf.float32)
epochs = 3 # howmany dataset is pushed trough network... This is the slow part
n_classes = 11
batch_size = 1 # howmany samples at once trough network.


def convolutional_neural_network(x):
    weights = {
        # 5 x 5 convolution, 1 input image, 32 outputs
        'W_conv1': tf.Variable(tf.random_normal([3, 3, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'W_conv2': tf.Variable(tf.random_normal([3, 3, 32, 64])),

        'W_conv3': tf.Variable(tf.random_normal([3, 3, 64, 128])),

        # fully connected, 7*7*64 inputs, 1024 outputs. 4*4 because after applying 3 convolution and max-pooling operations, you are downsampling the input image from 28 x 28 x 1 to 4 x 4 x 1.
        #'W_fc': tf.Variable(tf.random_normal([4*4*128, 1024])),
        'W_fc': tf.Variable(tf.random_normal([25 * 25 * 128, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_conv3': tf.Variable(tf.random_normal([128])),
        # 'b_fc': tf.Variable(tf.random_normal([1024])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Reshape input to a 4D tensor
    #used to be 28,28
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

    #fc = tf.reshape(conv3, [-1, 4 * 4 * 128])
    fc = tf.reshape(conv3, [-1, 25 * 25 * 128])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)


    output = tf.matmul(fc, weights['out']) + biases['out']
    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels=y)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer().minimize(cost) #default learning rate?


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

            #for b in range(int(mnist.train.num_examples / batch_size)):
            for b in range(int(data.num_train()/ batch_size)):

                #epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                epoch_x = train_X[b * batch_size:min((b + 1) * batch_size, len(train_X))]
                epoch_y = train_y[b * batch_size:min((b + 1) * batch_size, len(train_y))]



                opt = sess.run(optimizer, feed_dict={x: epoch_x, y: epoch_y})

                loss, acc = sess.run([cost, accuracy], feed_dict={x: epoch_x,
                                                                  y: epoch_y})


                #b, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                #epoch_loss += c

            print("epoch " + str(epoch) + ", Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            print("Optimization Finished!")

            # Calculate accuracy for all 10000 mnist test images
            test_accu, valid_loss = sess.run([accuracy, cost], feed_dict={x: epoch_x, y: epoch_y})
            train_loss.append(loss)
            test_loss.append(valid_loss)
            train_acc.append(acc)
            test_acc.append(test_accu)


        #print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        print('Accuracy:', accuracy.eval({x: data.test_x, y: data.test_y}))


train_neural_network(x)