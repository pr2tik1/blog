'''
Utility script for Doodle-Recognition

Link : https://pr2tik1.github.io/blog/pytorch/cnn/pca/t-sne/2020/09/08/Sketch-Recognition.html
'''

import os
import time
import math
import numpy as np
import urllib.request

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras 

import seaborn as sns
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from sklearn import decomposition
from sklearn import manifold
from sklearn import metrics

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

############################ Visualizations section ####################################
def get_pca(data, n_components = 2):
    """
    Function to perform PCA on dataset
    In: array, components
    Out: transformed array
    """
    pca = decomposition.PCA()
    pca.n_components = n_components
    pca_data = pca.fit_transform(data)
    return pca_data

def get_representations(model, iterator, device):
    """
    Evaulates through data loader and get representations
    In - Model, iterator/data loader, device=cpu/cuda
    Out -  Prediction output, Labels
    """
    model.eval()
    outputs = []
    labels = []

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y_pred = model(x)
            outputs.append(y_pred.cpu())
            labels.append(y)

    outputs = torch.cat(outputs, dim = 0)
    labels = torch.cat(labels, dim = 0)

    return outputs, labels

def plot_representations(data, labels, class_names, n_images = None):
    """
    Plots the predicted output and Labels with the chosen representation 
    TSNE/PCA.
    In - data, labels, list of Class Names, number of images
    Out - matplotlib figure
    """
    if n_images is not None:
        data = data[:n_images]
        labels = labels[:n_images]
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c = labels, cmap = 'tab10')
    handles, labels = scatter.legend_elements()
    legend = ax.legend(handles = handles, labels = class_names)

def get_tsne(data, n_components = 2, n_images = None):
    """
    Transforms data into TSNE representation
    In - data, number of components, number of images
    Out - TSNE transformed data
    """
    if n_images is not None:
        data = data[:n_images]
    tsne = manifold.TSNE(n_components = n_components, random_state = 0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data


############################# Model Section #############################
train_acc_list, train_loss_list = [], []
test_acc_list, test_loss_list = [], []

def count_parameters(model):
    """
    Counts the number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_accuracy(y_pred, y):
    """
    Calculates the accuracy of training/evaluating
    """
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, iterator, optimizer, criterion, device):  
    """
    Training function 
    Input : Model, Iterator = train data loader, Optimizer function,
            Criterian, device = cuda or cpu 
    Output: Training Loss and Training Accuracy
    """
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for (x, y) in iterator:
        x = x.float().to(device)
        y = y.type(torch.LongTensor).to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        
        acc = calculate_accuracy(y_pred, y)
        
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    train_acc_list.append(epoch_acc/len(iterator)) 
    train_loss_list.append(epoch_loss/len(iterator)) 
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator) , train_loss_list, train_acc_list

def evaluate(model, iterator, criterion, device):  
    """
    Evaluation Function
    Input : Model, iterator = test data loader, criterian of loss,
            device = cuda or cpu
    Output: Test loss, test accuracy
    """
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for (x, y) in iterator:
            x = x.float().to(device)
            y = y.type(torch.LongTensor).to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    test_acc_list.append(epoch_acc/len(iterator)) 
    test_loss_list.append(epoch_loss/len(iterator)) 

    return epoch_loss / len(iterator), epoch_acc / len(iterator) , test_loss_list, test_acc_list

def epoch_time(start_time, end_time):
    """
    Function to calculate total time taken in an epoch
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_predictions(model, iterator, device):
    """
    Function to fetch predictions
    Inputs: Model, test data loader, device = cuda/cpu
    Output: List of Images, Predicted labels and probability 
    """
    model.eval()
    images = []
    labels = []
    probs = []

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)

            y_pred = model(x)

            y_prob = F.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim = 0)
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)

    return images, labels, probs


################################# Data Section ###########################################


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


def load(valid_split = 0.2, max_items_per_class = 70000):
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
    
    x = np.reshape(x, (x.shape[0], 28, 28))

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
