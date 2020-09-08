import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras 

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import numpy as np
import os
import urllib.request

import seaborn as sns
from sklearn.manifold import TSNE


#Names of 10 Classes:
class_names = ['cloud', 'sun', 'pants', 'umbrella', 'table', 'ladder',
               'eyeglasses', 'clock', 'scissors', 'cup']

def download():
    """
    Function to download dataset from google api
    """
    root = 'data'
    os.mkdir('data')
    print('Downloading data with selected classes...')
    base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    for c in class_names:
        path = base+c+'.npy'
        print(path)
        urllib.request.urlretrieve(path, f'{root}/{c}.npy')
    print("Data Downloaded !")

def load(test_split = 0.2, max_items_per_class = 50000):
    """
    Function to load the dataset as train and test 
        - Generally : max_items_per_class = 10000
    """
    print("Loading data....")
    root = 'data'
    #initialize variables 
    x = np.empty([0, 784])
    y = np.empty([0])

    #load each data file 
    for idx, file in enumerate(class_names):
        data = np.load(f'{root}/{file}.npy')
        data = data[0: max_items_per_class, :]
        labels = np.full(data.shape[0], idx)

        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)

    data = None
    labels = None

    #randomize the dataset 
    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]

    #reshape and inverse the colors 
    x = 255 - np.reshape(x, (x.shape[0], 28, 28))

    #separate into training and testing 
    test_size  = int(x.shape[0]/100*(test_split*100))

    x_test = x[0:test_size, :]
    y_test = y[0:test_size]

    x_train = x[test_size:x.shape[0], :]
    y_train = y[test_size:y.shape[0]]

    print('Training Data : ', x_train.shape[0])
    print('Testing  Data : ', x_test.shape[0])
    return x_train, y_train, x_test, y_test, class_names




if __name__ == "__main__":
    download()
#    x_train, y_train, x_test, y_test, class_names = load()