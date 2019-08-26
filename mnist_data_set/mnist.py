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



class MNIST():

    URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    TRAIN_IMAGES_FILES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS_FILES = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGE_FILES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS_FILES = 't10k-labels-idx1-ubyte.gz'
    FILES_TO_GET = [TRAIN_IMAGES_FILES, TRAIN_LABELS_FILES, TEST_IMAGE_FILES, TEST_LABELS_FILES]
    IMAGE_SIZE = 28
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

    @staticmethod
    def download():
        for file_name in MNIST.FILES_TO_GET:
            MNIST._download_file(file_name)


def main():
    MNIST.download()

if __name__ == '__main__':
    main()

