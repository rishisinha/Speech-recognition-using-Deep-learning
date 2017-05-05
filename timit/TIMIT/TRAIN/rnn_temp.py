import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pandas as pd

features = np.asarray(pd.read_csv('inputs.csv'))
output_file = np.asarray(pd.read_csv('outputs.csv'))

inputs = features[:,2:]
labels = output_file[:,2:-1]

num_input = inputs.shape[0]
input_size = inputs.shape[1]
num_classes = labels.shape[1]

#print input_size

#num_layers = 2
num_epoch = 100
num_steps = 500
num_hidden = 512
batch_size = 25
learning_rate = 0.001
display_step = 5

x = tf.placeholder("float", [batch_size, num_steps, input_size])
y = tf.placeholder("float", [batch_size, num_steps, num_classes])

weights = {
    'out' : tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
}

biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def BiRNN(x, weights, biases):
    # reshape input to a list of nsteps tnsors of shape (batch_size, num_input) 
    x = tf.unstack(x, num_steps, 1)

    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias = 1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias = 1.0)

    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, \
                                                x, dtype = tf.float32)
    with tf.variable_scope('softmax'):
        logits = [tf.matmul(output, weights['out'])+biases['out'] for output in outputs]

    return logits

pred = BiRNN(x,weights,biases)

y_as_list = tf.unstack(y, num_steps, 1)
losses = []
cur_cor_pred = []
'''
for logit, label in zip(pred, y_as_list):
    losses.append(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(logit,1))
    cur_cor_pred.append(correct_pred)
'''
losses = [tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit) for \
            logit, label in zip(pred, y_as_list)]
cur_cor_pred = [ tf.equal(tf.argmax(label,1), tf.argmax(logit,1)) for \
            logit, label in zip(pred, y_as_list)]
total_loss = tf.reduce_mean(losses)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

cur_accuracy = tf.reduce_mean(tf.cast(cur_cor_pred, tf.float32))
num_cor_pred = tf.reduce_sum(tf.cast(cur_cor_pred, tf.float32))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    size = batch_size*num_steps
    
    for i in range(num_epoch):
        step = 1
        print 'Epoch ' + str(i)
        total_input = 0
        total_cor_pred = 0
        while step*size < num_input:
            #print step
            batch_x = inputs[(step-1)*size:step*size]
            batch_y = labels[(step-1)*size:step*size]
            total_input += step*size
            batch_x = batch_x.reshape(batch_size, num_steps, input_size)
            batch_y = batch_y.reshape(batch_size, num_steps, num_classes)
            sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})
            total_cor_pred += sess.run(num_cor_pred, feed_dict={x:batch_x, y:batch_y})
            if step % display_step == 0:
                cur_acc = sess.run(cur_accuracy, feed_dict={x:batch_x, y:batch_y})
                cur_loss = sess.run(total_loss, feed_dict={x:batch_x, y:batch_y})
                '''
                print 'Step ' + str(step) + '\n' + 'Current Loss ' + \
                    '{:.6f}'.format(cur_loss) + ', Training Accuracy= ' + \
                    '{:.5f}'.format(cur_acc)  
                '''
            step += 1
        print 'Total Accuracy in this epoch' + '{:.5f}'.format(total_cor_pred/total_input) 
print 'Done'

