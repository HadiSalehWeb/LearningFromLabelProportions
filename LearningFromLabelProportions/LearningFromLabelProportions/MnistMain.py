from ModelCreator import get_proportions_model
from ModelEvaluator import plot, show_images
from DataCleaner import get_processed_mnist_data
import keras.models as models

image_shape = (28, 28)
image_count = 20

data_train, labels_train, data_test, labels_test = get_processed_mnist_data(image_count, 1, 7)
model = get_proportions_model(image_shape + (1,), image_count)

data_train = data_train[0:600]
labels_train = labels_train[0:600]
data_test = data_test[0:600]
labels_test = labels_test[0:600]
labels_train = labels_train.reshape(labels_train.shape + (1,))
labels_test = labels_test.reshape(labels_test.shape + (1,))
history = model.fit(data_train, labels_train, 20, 5, 1, None, 0.2)

plot(history.history["acc"], history.history["val_acc"], 'Model Accuracy', 'Accuracy', 'Epoch')
plot(history.history["loss"], history.history["val_loss"], 'Model Loss', 'Loss', 'Epoch')

intermediate_layer_model = models.Model(inputs=model.input,outputs=model.get_layer('inter').output)
intermediate_output = intermediate_layer_model.predict(data_test)


show_images(data_test, labels_test, intermediate_output, 4, 5, image_shape)