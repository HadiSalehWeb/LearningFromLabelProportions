#import numpy as np
#import tensorflow as tf
#from tensorflow import keras
#import keras.layers as layers
#import keras.models as models
#import keras.optimizers as optimizers
#from keras.utils import plot_model
#from keras import backend as K
#from DataCleaner import get_processed_mnist_data
#import matplotlib.pyplot as plt

#image_shape = (28, 28, 1)
#image_count = 20

## Input > 2 conv layers > signle output
## Takes an image from the mnist dataset and spits out the probability of the image containing the number 1

#image_input = layers.Input(shape=image_shape)
#conv1 = layers.Convolution2D(5, 9, activation='relu', input_shape=image_shape)(image_input)
#conv2 = layers.Convolution2D(10, 5, activation='relu')(conv1)
#flat = layers.Flatten()(conv2)
#image_output = layers.Dense(1, activation='sigmoid')(flat)

#conv_model = models.Model(inputs=image_input, outputs=image_output)

#conv_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#conv_model.summary()

## input > TimeDistributed > average of the results of each element of the TimeSitributed layer > output, the average as a single number
## Takes in a group of images, processes them through the same neural network (the one we created above), and then averages the outputs for every image. Both the average and the singular outputs will be used in the training.

#set_input = layers.Input(shape=(image_count,) + image_shape)
#set_processing = layers.TimeDistributed(conv_model, name='inter')(set_input)
#set_output = layers.AveragePooling1D(image_count)(set_processing)
#set_flat_output = layers.Flatten()(set_output)

#set_model = models.Model(inputs=set_input, outputs=set_flat_output)

## Created and returns the lostt funtion
#def create_loss(layer, a=1):
#    # layer is the TimeDistributed layer, used here to extract the output from every image in the set.
#    # a is a hyperparameter that determines the importance of each term in the loss function
#    # (higher a value = more attention to the outputs of each network, lower value = more attention to the average value).
#    #@tf.function
#    def loss(y_true, y_pred):
#        loss_val = K.square(y_true - y_pred) + a * K.sum(layer * (1 - layer))
#        # K.square(y_true - y_pred): this makes the loss higher the further the average is from the correct proportion
#        # when the network is perfectly predicting the data, the TimeDistributed layer produces as many ones as there are images of ones in the set, same with zero. This makes the average the exact same as the proportion

#        # a * K.sum(layer * (1 - layer)): the loss is higher the more uncertain the first network is. This incentivises it to produce ones and zeroes instead of values between them.
#        return loss_val

#    return loss


#def custom_acc(y_true, y_pred):
#    return 10 - 10 * abs(y_true - y_pred)


## Arbitrary optimizer, I don't know if I should use a different one.
##set_model.compile(optimizer='adam', loss=create_loss(set_processing, 1 / image_count), metrics=['accuracy'])
##set_model.compile(optimizer='adam', loss=create_loss(set_processing, 0), metrics=['accuracy', custom_acc])
#set_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', custom_acc])

#set_model.summary()

##plot_model(conv_model, to_file='conv_model.png', show_shapes=True)
##plot_model(set_model, to_file='set_model.png', show_shapes=True)
##inp = np.random.rand(1, image_count, 28, 28, 1)
##print(set_model.predict(inp))

#data_train, labels_train, data_test, labels_test = get_processed_mnist_data(image_count,1,7)
#data_train = data_train[0:600]
#labels_train = labels_train[0:600]
#data_test = data_test[0:600]
#labels_test = labels_test[0:600]
#labels_train = labels_train.reshape(labels_train.shape + (1,))
#labels_test = labels_test.reshape(labels_test.shape + (1,))
#history = set_model.fit(data_train, labels_train, 20, 5, 1, None, 0.2)
    
## Plot training & validation accuracy values
#plt.plot(history.history['accuracy'])
#plt.title('Model accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()

## Plot training & validation loss values
#plt.plot(history.history['loss'])
#plt.title('Model loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()

#dim_1, dim_2 = 5, 4
#pred = set_model.predict(data_test)

#intermediate_layer_model = models.Model(inputs=set_model.input,
#                                 outputs=set_model.get_layer('inter').output)
#intermediate_output = intermediate_layer_model.predict(data_test)

##for j in range(len(data_test)):	
##    plt.figure()	
##    plt.suptitle('Proportion of ones: ' + str(labels_train[j]) + '. Predicted: ' + str(pred[j][0]))	
##    for i in range(dim_1 * dim_2):	
##        ax = plt.subplot(dim_1, dim_2, i + 1)	
##        ax.set_title(str(intermediate_output[j][i]))	
##        plt.imshow(np.reshape(data_test[j][i], (28, 28)), cmap=plt.cm.binary)	
##    plt.show()

#for j in range(len(data_test)):
#    s = 'true: ' + str(labels_test[j]) + ', m_odds: ' + str(np.mean(intermediate_output[j])) + '. m_th: ' + str(np.mean(intermediate_output[j]>0.5))
#    if(abs(np.mean(intermediate_output[j]>0.5)-labels_test[j])>0.05):
#        plt.figure()
#    #    plt.suptitle('Proportion of ones: ' + str(labels_train[j]) + '. Predicted: ' + str(pred[j][0]))
#        plt.suptitle(s)
    
#        for i in range(dim_1 * dim_2):
#            ax = plt.subplot(dim_1, dim_2, i + 1)
#            ax.set_title(str(intermediate_output[j][i]))
#            plt.imshow(np.reshape(data_test[j][i], (28, 28)), cmap=plt.cm.binary)
    
#        plt.show()
#    else:
#        print(s)