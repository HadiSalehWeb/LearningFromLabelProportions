import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers, models
import matplotlib.pyplot as plt

def get_processed_mnist_data(group_size, class0=0, class1=1):
    """Creates a set of sets, each containing a number of samples and the proportion of a certain label in that set. The propotion is the number of class1 items in the set, divided by the total number of items in the set.

    Keyword arguments:
    group_size -- the size of each set.
    class0 -- The filler label (default 0).
    class1 -- The measured label (default 0).
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    #print('x_train shape:', x_train.shape)
    #print(x_train.shape[0], 'train samples')
    #print(x_test.shape[0], 'test samples')

    # Separate the two classes from the dataset

    class0_train = np.array([x_train[i] for i in range(len(x_train)) if y_train[i] == class0])
    class1_train = np.array([x_train[i] for i in range(len(x_train)) if y_train[i] == class1])

    class0_test = np.array([x_test[i] for i in range(len(x_test)) if y_test[i] == class0])
    class1_test = np.array([x_test[i] for i in range(len(x_test)) if y_test[i] == class1])

    # Shuffle the labels from the two classes into the array of all labels
    
    shuffled_labels_train = np.array([class0 for i in range(len(class0_train))] + [class1 for i in range(len(class1_train))])
    shuffled_labels_test = np.array([class0 for i in range(len(class0_test))] + [class1 for i in range(len(class1_test))])
    np.random.shuffle(shuffled_labels_train)
    np.random.shuffle(shuffled_labels_test)

    # each label is assigned a sample from the dataset
    
    shuffled_data_train = []
    i0 = 0
    i1 = 0

    for i in shuffled_labels_train:
        if i == class0:
            shuffled_data_train.append(class0_train[i0])
            i0 += 1
        else:
            shuffled_data_train.append(class1_train[i1])
            i1 += 1

    shuffled_data_train = np.array(shuffled_data_train)

    shuffled_data_test = []
    i0 = 0
    i1 = 0

    for i in shuffled_labels_test:
        if i == class0:
            shuffled_data_test.append(class0_test[i0])
            i0 += 1
        else:
            shuffled_data_test.append(class1_test[i1])
            i1 += 1

    shuffled_data_test = np.array(shuffled_data_test)

    # The data is grouped into sets with the proportion of class_1 items in each set

    def group(data, labels, group_size, measured_label):
        return (np.array([data[i:i + group_size] for i in range(0, len(data), group_size) if i + group_size <= len(data)]),
                np.array([sum([1 for l in labels[i:i + group_size] if l == measured_label]) / group_size for i in range(0, len(data), group_size) if i + group_size <= len(data)]))

    final_data_train, final_labels_train = group(shuffled_data_train, shuffled_labels_train, group_size, class1)
    final_data_test, final_labels_test = group(shuffled_data_test, shuffled_labels_test, group_size, class1)
    
    # final_data shape: (groupCount, groupSize, 28: width, 28: height, 1) each element contains a pixel value between 0 and 1
    # final_labels shape: (groupCount,) each element contains a value between 0 and 1, the proportion of class1
    return final_data_train, final_labels_train, final_data_test, final_labels_test