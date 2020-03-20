from ModelCreator import get_proportions_model
from ModelEvaluator import plot, show_images
from CellData import get_cell_data
import keras.models as models

image_shape = (140, 140)
image_count = 100
batch_size = 10

data_train, labels_train = get_cell_data(image_count)
model = get_proportions_model(image_shape + (1,), image_count, 20, 10, 20, 10)

if len(data_train) % batch_size != 0:
    data_train = data_train[0:len(data_train) - (len(data_train) % batch_size)]
    labels_train = labels_train[0:len(labels_train) - (len(labels_train) % batch_size)]
    
labels_train = labels_train.reshape(labels_train.shape + (1,))
data_train = data_train.reshape(data_train.shape + (1,))

history = model.fit(data_train, labels_train, batch_size, 2, 1, None, 0.1)

plot(history.history["acc"], history.history["val_acc"], 'Model Accuracy', 'Accuracy', 'Epoch')
plot(history.history["loss"], history.history["val_loss"], 'Model Loss', 'Loss', 'Epoch')

intermediate_layer_model = models.Model(inputs=model.input,outputs=model.get_layer('inter').output)
intermediate_output = intermediate_layer_model.predict(data_train)


show_images(data_train, labels_train, intermediate_output, 2, 5, image_shape)