from ModelCreator import get_proportions_model
from ModelEvaluator import plot, show_images
from MnistDataProcessor import get_processed_mnist_data
import keras.models as models
import tensorflow as tf

tf.config.experimental.list_physical_devices('GPU')

image_shape = (28, 28)
bag_size = 20

# Get data
data_train, labels_train, data_test, labels_test = get_processed_mnist_data(bag_size, 1, 7)

#Create model
model = get_proportions_model(image_shape + (1,), bag_size)

# Round data size to batch size
data_train = data_train[0:600]
labels_train = labels_train[0:600]
data_test = data_test[0:600]
labels_test = labels_test[0:600]
labels_train = labels_train.reshape(labels_train.shape + (1,))
labels_test = labels_test.reshape(labels_test.shape + (1,))

# Train the model
history = model.fit(data_train, labels_train, 20, 5, 1, None, 0.2)

# Plot progression
plot(history.history["acc"], history.history["val_acc"], 'Model Accuracy', 'Accuracy', 'Epoch')
plot(history.history["loss"], history.history["val_loss"], 'Model Loss', 'Loss', 'Epoch')

# Get the single image prediction model
intermediate_layer_model = models.Model(inputs=model.input,outputs=model.get_layer('inter').output)
intermediate_output = intermediate_layer_model.predict(data_test)

# Predict single images and show result
show_images(data_test, labels_test, intermediate_output, 4, 5, image_shape)