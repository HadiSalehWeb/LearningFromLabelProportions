from ModelCreator import get_proportions_model
from ModelEvaluator import plot, show_images
from CustomDataProcessor import get_processed_data
import keras.models as models
import tensorflow as tf
import argparse
import os
import numpy as np

tf.config.experimental.list_physical_devices('GPU')

def train(directory, image_shape, proportions_path, bag_size, batch_size, filter1, kernel1, filter2, kernel2, epochs):
    # Get proportions
    proportions = np.loadtxt(proportions_path)

    # Get data
    data_train, labels_train = get_processed_data(directory, bag_size, proportions)

    #Create model
    model = get_proportions_model(image_shape + (1,), bag_size, filter1, kernel1, filter2, kernel2)

    # Round data size to batch size
    if len(data_train) % batch_size != 0:
        data_train = data_train[0:len(data_train) - (len(data_train) % batch_size)]
        labels_train = labels_train[0:len(labels_train) - (len(labels_train) % batch_size)]
    
    labels_train = labels_train.reshape(labels_train.shape + (1,))
    data_train = data_train.reshape(data_train.shape + (1,))

    # Train the model
    history = model.fit(data_train, labels_train, batch_size, epochs, 1, None, 0.1)

    # Plot progression
    plot(history.history["acc"], history.history["val_acc"], 'Model Accuracy', 'Accuracy', 'Epoch')
    plot(history.history["loss"], history.history["val_loss"], 'Model Loss', 'Loss', 'Epoch')

    # Get the single image prediction model
    intermediate_layer_model = models.Model(inputs=model.input,outputs=model.get_layer('inter').output)
    intermediate_output = intermediate_layer_model.predict(data_train)

    # Predict single images and show result
    show_images(data_train, labels_train, intermediate_output, 4, 5, bag_size)


def parse_tuple(str):
    return tuple(map(lambda str: int(str.strip()), str.split(',')))

def is_valid_path(arg):
    if not os.path.exists(arg):
        raise argparse.ArgumentTypeError('File %s does not exist.' % arg)
    else:
        return arg

def is_valid_data_path(arg):
    path = ''
    if '/' in arg:
        path = '/'.join(arg.split('/')[:-1])
    else:
        path = '\\'.join(arg.split('\\')[:-1])

    if not os.path.exists(path):
        raise argparse.ArgumentTypeError('File %s does not exist.' % path)
    else:
        return arg

parser = argparse.ArgumentParser(description='Trains a neural network to classify images based on a dataset of bag of those images along with their labels.')

parser.add_argument('-dir', dest='directory', help='path to the data directory, plus the shared initial name of the sub-directory names without the index. Defaults to "{current_dir}/data/tag_".', default=os.path.join(os.getcwd(), 'data', 'tag_'), type=is_valid_data_path)
parser.add_argument('-shape', dest='image_shape', help='width and height of one image. Defaults to (140, 140).', default=(140, 140), type=parse_tuple)
parser.add_argument('-prop', dest='proportions_path', help='path to the text file containing the proportion labels. Each line of the text file must contain on value. Defaults to "{current_dir}/data/labelproportions.txt".', default=os.path.join(os.getcwd(), 'data', 'labelproportions.txt'), type=is_valid_path)
parser.add_argument('-bag', dest='bag_size', help='Defaults to 100.', default=100, type=int)
parser.add_argument('-batch', dest='batch_size', help='Defaults to 1.', default=1, type=int)
parser.add_argument('-f1', dest='filter1', help='number of filters of the first convolutional layer. Defaults to 3.', default=3, type=int)
parser.add_argument('-k1', dest='kernel1', help='shape of filters of the first convolutional layer. Defaults to (50, 50).', default=(50, 50), type=parse_tuple)
parser.add_argument('-f2', dest='filter2', help='number of filters of the second convolutional layer. Defaults to 5.', default=5, type=int)
parser.add_argument('-k2', dest='kernel2', help='shape of filters of the second convolutional layer. Defaults to (10, 10).', default=(10,10), type=parse_tuple)
parser.add_argument('-epochs', dest='epochs', help='Defaults to 5.', default=5, type=int)

namespace = parser.parse_args()

train(namespace.directory, namespace.image_shape, namespace.proportions_path, namespace.bag_size, namespace.batch_size, namespace.filter1, namespace.kernel1, namespace.filter2, namespace.kernel2, namespace.epochs)
