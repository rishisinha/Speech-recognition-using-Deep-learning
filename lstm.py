import tensorflow as tf
import pandas as pd
import numpy as np

ip = pd.read_csv('inputs.csv')
op = pd.read_csv('outputs.csv')

inputs = np.asarray(ip.ix[:,2:])
labels_ = np.asarray(op.ix[:,2:-1])

labels = np.zeros((labels_.shape[0],1))

for i in range(labels.shape[0]):
    labels[i,0] = np.argmax(labels_[i,:])

epochs = 100
num_classes = labels_.shape[1]
batch_size = 5
features = inputs.shape[1]
hidden = 128
learning_rate = 0.001
seq_max_len = 20
training_iters = 100000

x = tf.placeholder(tf.float32, shape = [None, seq_max_len, features])
y = tf.placeholder(tf.float32, shape = [None, num_classes])

seqlen = tf.placeholder(tf.int32, [None])

weights = {'out': tf.Variable(tf.random_normal([hidden, num_classes]))}

biases = { 'out': tf.Variable(tf.random_normal([num_classes]))}

def dynamicRNN(x, seqlen, weights, biases):

    x = tf.unstack(x, seq_max_len, 1)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])
    batch_size = tf.shape(outputs)[0]
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, hidden]), index)
    return tf.matmul(outputs, weights['out']) + biases['out']

pred = dynamicRNN(x, seqlen, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
