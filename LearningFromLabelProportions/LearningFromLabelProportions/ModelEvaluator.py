import numpy as np
import matplotlib.pyplot as plt


def plot(data_train, data_test, title, ylabel, xlabel):
    plt.plot(data_train)
    plt.plot(data_test)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def show_images(data, labels, pred, dim1, dim2, image_shape):
    for j in range(len(data)):
        plt.figure()
        plt.suptitle('true: ' + str(labels[j]) + 
            ', m_odds: ' + str(np.mean(pred[j])) + 
            '. m_th: ' + str(np.mean(pred[j]>0.5)))

        for i in range(dim1 * dim2):
            ax = plt.subplot(dim1, dim2, i + 1)
            ax.set_title(str(pred[j][i]))
            plt.imshow(np.reshape(data[j][i], image_shape), cmap=plt.cm.binary)
    
        plt.show()