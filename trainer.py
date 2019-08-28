#!c\\programy\\Anaconda3\\envs\\tensorflow\\python

'''
copyright (c) 2019 DATAUP Mateusz Kotarski 2019
All rights reserved
'''
import os 

from mnist_data_set.mnist import MNIST
import tensorflow as tf

data = tf.placeholder(tf.float32, [None, MNIST.IMAGE_SIZE **2])
labels = tf.placeholder(tf.float32, [None, 10])
weights = tf.Variable(tf.zeros([ MNIST.IMAGE_SIZE ** 2, 10]))
biases = tf.Variable(tf.zeros([10]))
logits = tf.matmul(data, weights) + biases
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, axis=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                           labels=labels)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

def main():
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    batch_size = 100
    mnist_set = MNIST()
    
    for i in range(100):
        batch_data, batch_labels = mnist_set.get_train_batch(batch_size)
        batch_labels = mnist_set.one_hot_encoded(batch_labels, 10)
        feed_dict_train = {data: batch_data,
                           labels: batch_labels}
        session.run(optimizer, feed_dict=feed_dict_train)
        temp_logits = session.run(logits, feed_dict=feed_dict_train)
        for idx, nic in enumerate(temp_logits):
            print(nic)
            print(batch_labels[idx])
            print('*****')


if __name__ == '__main__':
    main()
