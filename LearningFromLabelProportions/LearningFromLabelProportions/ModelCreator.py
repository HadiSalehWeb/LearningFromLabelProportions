from tensorflow import keras
import keras.layers as layers
import keras.models as models

def get_proportions_model(image_shape,image_count,filter1=5,kernel1=9,filters2=10,kernel2=5):

    # Input > 2 conv layers > single output
    # Takes an image from the dataset and spits out the probability of the image containing the measured label

    image_input = layers.Input(shape=image_shape)
    conv1 = layers.Convolution2D(5, 9, activation='relu', input_shape=image_shape)(image_input)
    conv2 = layers.Convolution2D(10, 5, activation='relu')(conv1)
    flat = layers.Flatten()(conv2)
    image_output = layers.Dense(1, activation='sigmoid')(flat)

    conv_model = models.Model(inputs=image_input, outputs=image_output)

    conv_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    #conv_model.summary()

    # input > TimeDistributed > average of the results of each element of the TimeSitributed layer > output, the average as a single number
    # Takes in a group of images, processes them through the same neural network (the one we created above), and then averages the outputs for every image. Both the average and the singular outputs will be used in the training.

    set_input = layers.Input(shape=(image_count,) + image_shape)
    set_processing = layers.TimeDistributed(conv_model, name='inter')(set_input)
    set_output = layers.AveragePooling1D(image_count)(set_processing)
    set_flat_output = layers.Flatten()(set_output)

    set_model = models.Model(inputs=set_input, outputs=set_flat_output)

    # Arbitrary optimizer, I don't know if I should use a different one.
    #set_model.compile(optimizer='adam', loss=create_loss(set_processing, 1 / image_count), metrics=['accuracy'])
    set_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return set_model