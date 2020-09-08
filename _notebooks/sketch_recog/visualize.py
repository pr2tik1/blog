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

def plot_confusion_matrix(labels, pred_labels):
    """Plots Confusion Matrix"""
    fig = plt.figure(figsize = (10, 10));
    ax = fig.add_subplot(1, 1, 1);
    cm = metrics.confusion_matrix(labels, pred_labels);
    cm = metrics.ConfusionMatrixDisplay(cm, range(10));
    cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
