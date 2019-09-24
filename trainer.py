#!c\\programy\\Anaconda3\\envs\\tensorflow\\python

'''
copyright (c) 2019 DATAUP Mateusz Kotarski 2019
All rights reserved
'''
import os 
import numpy as np
from mnist_data_set.mnist import MNIST
import tensorflow as tf

from models import ModelSimple
import copy

def train_net_self_teacher_target_label(target_label, data_set):
    batch_size = 100
    change = False
    model = ModelSimple().from_scratch(num_outputs = 20, learning_rate = 0.3)
    iter_num = 0
    set_ = 0
    while True:
        batch_data, batch_labels = data_set.get_train_batch_from_set(batch_size, set_)
        batch_labels_encoded = data_set.one_hot_encoded(batch_labels, 20)
        model.train_on_batch(batch_data, batch_labels_encoded)
        if iter_num >= data_set.get_num_train_samples() * 10 and not change:
            change = True
        if iter_num >= data_set.get_num_train_samples() * 10  and change:
            iter_num = 0
            test_data, test_labels = data_set.get_test_set()
            test_labels_encoded = data_set.one_hot_encoded(test_labels, 20)
            best_score = model.validate_self_teacher(test_data, test_labels_encoded , target_label)
            model.save_model(target_label)
            exit()
            if set_ == 0:
                test_labels = data_set.one_hot_encoded(data_set.nn_train_lables_0, 20)
                new_labels = model.get_new_labels(data_set.nn_train_data_0, test_labels, target_label)
                data_set.nn_train_lables_0 = new_labels
                set_ = 1
            if set == 1:
                test_labels = data_set.one_hot_encoded(data_set.nn_train_lables_1, 20)
                new_labels = model.get_new_labels(data_set.nn_train_data_1, test_labels, target_label)
                data_set.nn_train_lables_1 = new_labels
                set_ = 0
        iter_num += batch_size


def main():
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    mnist_set = MNIST()
    for i in range(0, 10):
        print("Running training for label: %d" %(i))
        mnist_set = MNIST()
        train_net_self_teacher_target_label(i, mnist_set)

        
if __name__ == '__main__':
    main()
