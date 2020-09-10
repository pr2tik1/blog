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


def load(valid_split = 0.2, max_items_per_class = 50000):
    """
    Function to load the dataset as train and test 
        - Generally : max_items_per_class = 10000
    """
    print("Loading data....")
    root = 'data'
    #initialize variables 
    x = np.empty([0, 784])
    y = np.empty([0])

    x_test = np.empty([0, 784])
    y_test = np.empty([0])

    #load each data file 
    for idx, file in enumerate(class_names):
        data = np.load(f'{root}/{file}.npy')
        
        test_data = data[max_items_per_class:max_items_per_class+10000,:] 
        data = data[0: max_items_per_class, :]
        
        labels = np.full(data.shape[0], idx)
        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)
        
        test_labels = np.full(test_data.shape[0], idx)
        x_test = np.concatenate((x_test, test_data), axis=0)
        y_test = np.append(y_test, test_labels)

    data = None
    labels = None
    test_data = None
    test_labels = None

    #randomize the dataset 
    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]

    permutation_1 = np.random.permutation(y_test.shape[0])
    x_test = x_test[permutation_1, :]
    y_test = y_test[permutation_1]
    
    #reshape and inverse the colors 
    x_test = np.reshape(x_test, (x_test.shape[0], 28, 28))
    x_test = np.reshape(x_test, (x_test.shape[0], 28, 28))
    
    x = 255 - np.reshape(x, (x.shape[0], 28, 28))

    #separate into training and testing 
    valid_size  = int(x.shape[0]/100*(valid_split*100))

    x_valid = x[0:valid_size, :]
    y_valid = y[0:valid_size]

    x_train = x[valid_size:x.shape[0], :]
    y_train = y[valid_size:y.shape[0]]

    print('Training Data : ', x_train.shape[0])
    print('Validating  Data : ', x_valid.shape[0])
    print('Testing  Data : ', x_test.shape[0])
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test, class_names




if __name__ == "__main__":
    download()
    #collapse-hide
    #x_train, y_train, x_valid, y_valid, test_data, test_labels, class_names = load()
    #print("\nShape of Training data(X,y): ", x_train.shape, y_train.shape)
    #print("Shape of Validating data(X,y): ", x_valid.shape, y_valid.shape)
    #print("Shape of Testing data(X,y): ", test_data.shape)
    #print("Classes : ",class_names) 