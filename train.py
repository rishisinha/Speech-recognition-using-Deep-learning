import tensorflow as tf
import pandas as pd
import numpy as np

inputs = pd.read_csv('inputs.csv')
outputs = pd.read_csv('outputs.csv')


num_feats = 13


batch_size = 1
num_classes = outputs.shape[1] - 3
epochs = 200
alpha = 0.001
lstm_size = 50
number_of_layers = 1
num_hidden = 50

input_temp = np.asarray(inputs.ix[:,2:])
input_temp = input_temp[np.newaxis,:]
train_seq_len = np.array([input_temp.shape[1]])
out_temp = np.asarray(outputs.ix[:,2:-1])

labels = np.zeros((out_temp.shape[0],1))

for i in range(labels.shape[0]):
    labels[i,0] = np.argmax(out_temp[i,:])

graph = tf.Graph()

with graph.as_default():

    ip = tf.placeholder(tf.float32, shape = [None, None, num_feats])

    op = tf.sparse_placeholder(tf.int32,[None])

    seq_len = tf.placeholder(tf.int32, [None])

    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * number_of_layers, state_is_tuple=True)
    
    out, out_states = tf.nn.dynamic_rnn(lstm, ip, seq_len, dtype=tf.float32)
    shape = tf.shape(ip)
    batch_s, max_timesteps = shape[0], shape[1]

    out = tf.reshape(out, [-1, num_hidden])

    w = tf.Variable(tf.truncated_normal([num_hidden,num_classes], stddev = 0.1))

    b = tf.Variable(tf.constant(0, shape = [num_classes], dtype=tf.float32))
    
    mul = tf.matmul(out,w) + b

    mul = tf.reshape(mul, [batch_s, -1, num_classes])

    mul = tf.transpose(mul, [1,0,2])

    loss = tf.nn.ctc_loss(op, mul, seq_len)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.MomentumOptimizer(alpha, 0.9).minimize(cost)

    decoded, log_prob = tf.nn.ctc_greedy_decoder(mul, seq_len)

    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), op))
with tf.Session(graph=graph) as session:

    tf.global_variables_initializer().run()

    for epoch in range(epochs):
        train_cost = 0
        train_ler = 0
        feed = {ip: input_temp, op: labels, seq_len: train_seq_len}
        batch_cost, _ = session.run([cost, optimizer], feed)
        train_cost += batch_cost*batch_size
        train_ler += session.run(ler, feed_dict=feed)*batch_size
        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler))
