#!c\\programy\\Anaconda3\\envs\\tensorflow\\python

'''
copyright (c) 2019 DATAUP Mateusz Kotarski 2019
All rights reserved
'''
import urllib.request
import shutil
import numpy as np
import gzip
import os
import matplotlib.pyplot as plt 


class MNIST():

    URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    TRAIN_IMAGES_FILES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS_FILES = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGE_FILES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS_FILES = 't10k-labels-idx1-ubyte.gz'
    FILES_TO_GET = [TRAIN_IMAGES_FILES, TRAIN_LABELS_FILES, TEST_IMAGE_FILES, TEST_LABELS_FILES]
    IMAGE_SIZE = 28
    OFFSET_IMAGES = 16
    OFFSET_LABELS = 8
    #only one colour chanell
    SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)

    @staticmethod
    def _download_file(file_name):
        file = (os.path.splitext(file_name))[0]
        if os.path.isfile(file):
            return
        urllib.request.urlretrieve(filename = file_name, url = MNIST.URL + file_name)
        with gzip.open(file_name, 'rb') as f_in:
            with open(file, 'wb') as f_out: # remove .gz
                shutil.copyfileobj(f_in, f_out)


    def _load_data(self, file_name, offset):
        mnist_dir = os.path.dirname(os.path.abspath(__file__))
        file = (os.path.join(mnist_dir, os.path.splitext(file_name)[0]))
        if not os.path.isfile(file):
            MNIST.download(file_name)
        with open(file, 'rb') as file_handler:
            data = np.frombuffer(file_handler.read(), np.uint8, offset = offset)
        return data


    @staticmethod
    def download():
        for file_name in MNIST.FILES_TO_GET:
            MNIST._download_file(file_name)

    def plot_images(self):
        fig, axes = plt.subplots(3, 3)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        for i, ax in enumerate(axes.flat):
            # Plot image.
            ax.imshow(self.train_data[i+12].reshape((self.IMAGE_SIZE, self.IMAGE_SIZE)), cmap='binary')
        '''
        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        
        ax.set_xlabel(xlabel)
        '''
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
        plt.show()

    def get_train_batch(self, batch_size):
        idx = np.random.randint(low=0, high=self.train_data.shape[0], size=batch_size)
        return self.train_data[idx], self.train_lables[idx] 

    #TODO: move to utils functions
    @staticmethod
    def one_hot_encoded(class_numbers, num_classes=None):
        if num_classes is None:
            num_classes = np.max(class_numbers) + 1

        return np.eye(num_classes, dtype=float)[class_numbers]


    def __init__(self):
        self.train_data = self._load_data(MNIST.TRAIN_IMAGES_FILES, MNIST.OFFSET_IMAGES)
        self.train_data  = self.train_data.reshape(-1, self.IMAGE_SIZE ** 2)/255
        self.train_lables = self._load_data(MNIST.TRAIN_LABELS_FILES, MNIST.OFFSET_LABELS)


def main():
    test = MNIST()
 

if __name__ == '__main__':
    main()

