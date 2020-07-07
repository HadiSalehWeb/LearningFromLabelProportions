from ModelCreator import get_proportions_model
from ModelEvaluator import plot, show_images
from CellDataProcessor import get_cell_data
import keras.models as models
import tensorflow as tf

tf.config.experimental.list_physical_devices('GPU')

image_shape = (140, 140)
bag_size = 800
batch_size = 1

# Get data
data_train, labels_train = get_cell_data(bag_size)

#Create model
model = get_proportions_model(image_shape + (1,), bag_size, 3, 50, 5, 10)

# Round data size to batch size
if len(data_train) % batch_size != 0:
    data_train = data_train[0:len(data_train) - (len(data_train) % batch_size)]
    labels_train = labels_train[0:len(labels_train) - (len(labels_train) % batch_size)]
    
labels_train = labels_train.reshape(labels_train.shape + (1,))
data_train = data_train.reshape(data_train.shape + (1,))

# Train the model
history = model.fit(data_train, labels_train, batch_size, 2, 1, None, 0.1)

# Plot progression
plot(history.history["acc"], history.history["val_acc"], 'Model Accuracy', 'Accuracy', 'Epoch')
plot(history.history["loss"], history.history["val_loss"], 'Model Loss', 'Loss', 'Epoch')

# Get the single image prediction model
intermediate_layer_model = models.Model(inputs=model.input,outputs=model.get_layer('inter').output)
intermediate_output = intermediate_layer_model.predict(data_train)

# Predict single images and show result
show_images(data_train, labels_train, intermediate_output, 4, 5, bag_size)