import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers, models
import matplotlib.pyplot as plt

def get_processed_mnist_data(group_size):
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

    # Separate zeroes and ones

    zeroes_train = np.array([x_train[i] for i in range(len(x_train)) if y_train[i] == 0])
    ones_train = np.array([x_train[i] for i in range(len(x_train)) if y_train[i] == 1])

    zeroes_test = np.array([x_test[i] for i in range(len(x_test)) if y_test[i] == 0])
    ones_test = np.array([x_test[i] for i in range(len(x_test)) if y_test[i] == 1])

    # The labels are an array of shuffled 0's and 1's
    
    shuffled_labels_train = np.array([0 for i in range(len(zeroes_train))] + [1 for i in range(len(ones_train))])
    shuffled_labels_test = np.array([0 for i in range(len(zeroes_test))] + [1 for i in range(len(ones_test))])
    np.random.shuffle(shuffled_labels_train)
    np.random.shuffle(shuffled_labels_test)

    # each zero and one is assigned a sample from the dataset
    
    shuffled_data_train = []
    i_zero = 0
    i_one = 0

    for i in shuffled_labels_train:
        if i == 0:
            shuffled_data_train.append(zeroes_train[i_zero])
            i_zero += 1
        else:
            shuffled_data_train.append(ones_train[i_one])
            i_one += 1

    shuffled_data_train = np.array(shuffled_data_train)

    shuffled_data_test = []
    i_zero = 0
    i_one = 0

    for i in shuffled_labels_test:
        if i == 0:
            shuffled_data_test.append(zeroes_test[i_zero])
            i_zero += 1
        else:
            shuffled_data_test.append(ones_test[i_one])
            i_one += 1

    shuffled_data_test = np.array(shuffled_data_test)

    # The data is grouped into sets with the proportion of 1's in each set

    def group(data, labels, group_size, measured_label):
        return (np.array([data[i:i + group_size] for i in range(0, len(data), group_size) if i + group_size <= len(data)]),
                np.array([sum([1 for l in labels[i:i + group_size] if l == measured_label]) / group_size for i in range(0, len(data), group_size) if i + group_size <= len(data)]))

    #dim_1, dim_2 = 5, 4
    final_data_train, final_labels_train = group(shuffled_data_train, shuffled_labels_train, group_size, 1)
    final_data_test, final_labels_test = group(shuffled_data_test, shuffled_labels_test, group_size, 1)
    
    # final_data shape: (groupCount, groupSize, 28: width, 28: height, 1) each element contains a pixel value between 0 and 1
    # final_labels shape: (groupCount,) each element contains a value between 0 and 1, the proportion of 1's
    return final_data_train, final_labels_train, final_data_test, final_labels_test


    #for j in range(len(final_data)):
    #    plt.figure()
    #    plt.suptitle('Proportion of ones: ' + str(final_labels[j]))

    #    for i in range(dim_1 * dim_2):
    #        plt.subplot(dim_1, dim_2, i + 1)
    #        plt.imshow(np.reshape(final_data[j][i], (28, 28)), cmap=plt.cm.binary)

    #    plt.show()

    #np.array([a[i:i + group_size] for i in range(0, len(a), 2)])