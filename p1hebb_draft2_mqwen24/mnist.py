'''mnist.py
Loads and preprocesses the MNIST dataset
MUQING WEN, ZHOUYI QIAN
CS443: Bio-Inspired Machine Learning
Project 1: Hebbian Learning
'''
import os
import numpy as np


def get_mnist(N_val, path='data/mnist'):
    '''Load and preprocesses the MNIST dataset (train and test sets) located on disk within `path`.

    Parameters:
    -----------
    N_val: int. Number of data samples to reserve from the training set to form the validation set. As usual, each
    sample should be in EITHER training or validation sets, NOT BOTH.
    path: str. Path in working directory where MNIST dataset files are located.

    Returns:
    -----------
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels)
    '''
    path_1 = os.path.join(path, "x_train.npy")
    path_2 = os.path.join(path, "y_train.npy")
    path_3 = os.path.join(path, "x_test.npy")
    path_4 = os.path.join(path, "y_test.npy")
    
    x_train = np.load(path_1)
    y_train = np.load(path_2)
    x_test = np.load(path_3)
    y_test = np.load(path_4)
    
    x_train = np.reshape(x_train, (x_train.shape[0], np.prod(x_train.shape[1:])))
    x_test = np.reshape(x_test, (x_test.shape[0], np.prod(x_test.shape[1:])))
    
    x_train = x_train / 255
    x_test = x_test / 255
    
    val_indx = np.random.choice(a=60000, size=100, replace=False)
    
    x_val = x_train[val_indx].copy()
    y_val = y_train[val_indx].copy()
    
    x_train = np.delete(x_train, val_indx, axis=0)
    y_train = np.delete(y_train, val_indx, axis=0)
    
    return x_train, y_train, x_test, y_test, x_val, y_val


def preprocess_mnist(x):
    '''Preprocess the data `x` so that:
    - the maximum possible value in the dataset is 1 (and minimum possible is 0).
    - the shape is in the format: `(N, M)`

    Parameters:
    -----------
    x: ndarray. shape=(N, I_y, I_x). MNIST data samples represented as grayscale images.

    Returns:
    -----------
    ndarray. shape=(N, I_y*I_x). MNIST data samples represented as MLP-compatible feature vectors.
    '''
    pass


def train_val_split(x, y, N_val):
    '''Divide samples into train and validation sets. As usual, each sample should be in EITHER training or validation
    sets, NOT BOTH. Data samples are already shuffled.

    Parameters:
    -----------
    x: ndarray. shape=(N, M). MNIST data samples represented as vector vectors
    y: ndarray. ints. shape=(N,). MNIST class labels.
    N_val: int. Number of data samples to reserve from the training set to form the validation set.

    Returns:
    -----------
    x: ndarray. shape=(N-N_val, M). Training set.
    y: ndarray. shape=(N-N_val,). Training set class labels.
    x_val: ndarray. shape=(N_val, M). Validation set.
    y_val ndarray. shape=(N_val,). Validation set class labels.
    '''
    pass