#!c\\programy\\Anaconda3\\envs\\tensorflow\\python

'''
copyright (c) 2019 DATAUP Mateusz Kotarski 2019
All rights reserved
'''

import os
import tensorflow as tf
import numpy as np
import xgboost as xgb

from mnist_data_set.mnist import MNIST
tf.set_random_seed(0)


class XgbModel(object):

    def __init__(self):
        pass

    def train(self):
        pass

class XGBModelSimple(XgbModel):
    
    def __init__(self):
        self.model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)

    def train(self, data, labels):
        self.model.fit(data, labels)

    def predict(self, data, labels):
        self.model.predict(data)


class MnistModel(object):
    MODEL_TYPE = 'NN'
    NAME = "?"
    def __init__(self):
        self.components = dict()
        self.components['input'] = None
        self.components['input_labels'] = None
        self.components['logits'] = None
        self.components['soft_max'] = None
        self.components['loss_function'] = None
        self.components['optimizer'] = None
        self.session  = None
        self.iter = 0

    def train_on_batch(self, batch_data, batch_labels):
        self.session.run(self.components['optimizer'], {self.components['input']: batch_data,  self.components['input_labels'] : batch_labels})

    def before_back_propagation(self, data, data_labels):
        cross_entropy, after_softmax = self.session.run([self.components['logits'],  self.components['soft_max']], {self.components['input']: data,  self.components['input_labels'] : data_labels})
        return cross_entropy, after_softmax

    def get_new_labels(self, data, data_labels, target_label):
        labels = list()
        after_softmax, _ = self.before_back_propagation(data, data_labels)
        for predictions, labels_encoded in zip(after_softmax, data_labels):
            normal_output = predictions[:10] + predictions[10:] 
            prediction = np.where(normal_output == np.amax(normal_output))[0][0]
            correct_label = np.where(labels_encoded == np.amax(labels_encoded))[0][0]
            if correct_label != target_label and prediction != target_label:
                labels.append(prediction)
            #false negative
            if correct_label == target_label and prediction != target_label:
                labels.append(target_label)
            #false positive
            if correct_label != target_label and prediction == target_label:

                labels.append(correct_label + 10) #?? 11
        if correct_label == target_label and prediction == target_label:
            labels.append(target_label) #?? 11
        return labels



    #TODO: move to utils and add description
    def validate_self_teacher(self, data, data_labels, target_label):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        i = 0
        after_softmax, _ = self.before_back_propagation(data, data_labels)
        for predictions, labels_encoded in zip(after_softmax, data_labels):
            normal_output = predictions[:10] + predictions[10:]
            correct_label = np.where(labels_encoded == np.amax(labels_encoded))[0][0]
            prediction = np.where(normal_output == np.amax(normal_output))[0][0]
            #false positive
            if correct_label != target_label and prediction == target_label:
                false_positives += 1
            if correct_label == target_label and prediction == target_label:
                true_positives += 1
            #false_negative
            if correct_label == target_label and prediction != target_label:
                false_negatives += 1
                i += 1
        try:
            precision = true_positives/(true_positives + false_positives)
        except ZeroDivisionError:
            precision = 0
    
        try:
            recall = true_positives/(true_positives + false_negatives)
        except ZeroDivisionError:
            recall = 0
        
        try:
            F_score = 2*(precision*recall)/(precision+recall)
        except ZeroDivisionError:
            F_score = 0
    
        print('Stats for label:  %f'%(target_label))
        print('true_positives:  %f'%(true_positives))
        print('false_positives:  %f'%(false_positives))
        print('false_negatives:  %f'%(false_negatives))
        print('F score:  %f'%(F_score))
        print('******')
        return F_score

    def save_model(self, model_description = ''):
        saver = tf.train.Saver()
        saver.save(self.session, os.path.join(os.path.realpath(__file__),os.pardir,"trained_models", self.MODEL_TYPE + self.NAME + str(model_description)))

    def train(self):
        pass

    def get_session(self):
        return self.sess

class ModelSimple(MnistModel):
    NAME = "SIMPLE_VANILA"

    @classmethod
    def from_file(cls):
        pass

    @classmethod
    def from_scratch(cls, num_outputs, learning_rate):
        model = cls()
        model.components['input'] = tf.placeholder(tf.float32, [None, MNIST.IMAGE_SIZE ** 2], name='input')
        model.components['input_labels'] = tf.placeholder(tf.float32, [None, num_outputs], name='input_labels')
        #layer 1
        model.components['weights_1'] = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1), name = 'weights_1')
        model.components['biases_1'] = tf.Variable(tf.zeros([200]), name = 'biases_1')
        #layer 2
        model.components['weights_2'] = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1), name = 'weights_2')
        model.components['biases_2'] = tf.Variable(tf.zeros([100]), name = 'biases_2')
        #layer 3
        model.components['weights_3'] = tf.Variable(tf.truncated_normal([100, 60], stddev=0.1), name = 'weights_3')
        model.components['biases_3'] = tf.Variable(tf.zeros([60]), name = 'biases_3')
        #layer 4
        model.components['weights_4'] = tf.Variable(tf.truncated_normal([60, 50], stddev=0.1), name = 'weights_4')
        model.components['biases_4'] = tf.Variable(tf.zeros([50]), name = 'biases_4')
        #layer 5
        model.components['weights_5'] = tf.Variable(tf.truncated_normal([50, num_outputs], stddev=0.1), name = 'weights_5')
        model.components['biases_5'] = tf.Variable(tf.zeros([num_outputs]), name = 'biases_5')
        #operations
        layer_1_output = tf.nn.sigmoid(tf.matmul(model.components['input'], model.components['weights_1']) + model.components['biases_1'])
        layer_2_output = tf.nn.sigmoid(tf.matmul(layer_1_output, model.components['weights_2']) + model.components['biases_2'])
        layer_3_output = tf.nn.sigmoid(tf.matmul(layer_2_output, model.components['weights_3']) + model.components['biases_3'])
        layer_4_output = tf.nn.sigmoid(tf.matmul(layer_3_output, model.components['weights_4']) + model.components['biases_4'])
        model.components['logits'] = tf.matmul(layer_4_output, model.components['weights_5']) +  model.components['biases_5']
        model.components['soft_max'] =  tf.nn.softmax(model.components['logits'])
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.components['logits'], labels=model.components['input_labels'])
        cross_entropy = tf.reduce_mean(cross_entropy)
        model.components['loss_function'] = tf.reduce_mean(cross_entropy)
        model.components['optimizer'] =  tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(model.components['loss_function'])
        init = tf.global_variables_initializer()
        model.session = tf.Session()
        model.session.run(init)
        return model

    def __init__(self):
        super().__init__()

