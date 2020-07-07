from ModelCreator import get_proportions_model
from ModelEvaluator import plot, show_images
from CustomDataProcessor import get_processed_data
import keras.models as models
import tensorflow as tf

tf.config.experimental.list_physical_devices('GPU')

print("~~~Intorduction~~~")
print("To run the program on a custom dataset, you need to create a directory with sub-directories under it, each containing a group of images. You will be asked to input several pieces of information about these directories and images.")
print("At the end after the model is trained, a graph of the loss function and accuracy over epochs will be displayed. After closing the graph windows, another window will open with 20 samples of the predictions of the trained network. Closing that window will display a different sample, and so on.")
print("Note regarding directory structure: give the main directory an arbitrary name e.g 'fermentation1', then give all the sub-directories under it a unified name with a different index at the end, e.g. 'fermentation1/batch_0', 'fermentation1/batch_1' etc. The indices must be consecutive and start at 0. When asked to input the directory path, provide both the path to the main directory as well as the name of the sub-directories without the index, e.g. 'D:/fermentation1/batch_'.")

directory = str(input("Please enter the directory path: "))
image_shape = tuple(map(lambda str: int(str.strip()), input("Enter the width and height of one image as two comma separated integers Exmaple: 16, 16. Input: ").split(',')))
proportions = list(map(lambda str: float(str.strip()), input("Enter the proportion of the measured label in each subdirectory in order, as a list of comma separated numbers between 0 and 1. Example: 0.5, 0.2, 0.75. Input: ").split(',')))
print("Hyperparameters: ")
bag_size = int(input("Enter the size of one bag (custom value, the larger this number the more accurate the labels and therefore the better the prediction but the more time one learning iteration takes. Has to be a factor of the number of images in each directory, otherwise the numbers will be rounded and some data might be discarded): "))
batch_size = int(input("Enter the batch size (the number of bags processed on each learning iteration, select small number for large bag size). Enter nothing to set it to default (1): ") or 1)
filter1 = int(input("Enter the number of filters on the first convolutional layer. Enter nothing for default (3): ") or 3)
kernel1 = tuple(map(lambda str: int(str.strip()), (input("Enter the width and height of the convolutional window of the first layer as two comma separated integers. Enter nothing for default (50, 50): ") or "50, 50").split(',')))
filter2 = int(input("Enter the number of filters on the second convolutional layer. Enter nothing for default (5): ") or 5)
kernel2 = tuple(map(lambda str: int(str.strip()), (input("Enter the width and height of the convolutional window of the seconod layer as two comma separated integers. Enter nothing for default (10, 10): ") or "10, 10").split(',')))
epochs = int(input("Enter the number of epochs (how many times the model will be trained on the same dataset). Higher number means more time but potentially more accurate predictions. Enter nothing for default (5): ") or "5")

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