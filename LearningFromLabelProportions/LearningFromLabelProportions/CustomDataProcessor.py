import matplotlib.pyplot as plt
import imageio
import numpy as np
import os
import os.path

def get_processed_data(directory, bag_size, proportions):
    """Creates a set of bags, each containing a number of samples and the proportion of the measured label in that bag.

    Directory example:
    .
    +-- D:
        +-- Fermentation_1
            +-- Bag_1
            +-- Bag_2
            +-- Bag_3
    
    In this example, the directory parameter would be "D:/Fermentation_1/Bag_"
    
    Keyword arguments:
    directory: The path to the directories with the data samples, plus the name of the sub-directories WITHOUT their index.
    bag_size: the size of each bag.
    proportions: A list of the proportions of each bag in the directory, each bag being represented by another sub-directory.
    """
    
    count = len(proportions)
    arr = [[] for i in range(count)]
    j = 0
    print("Loading data #: ", end="")

    for i in range(count):
        print(str(i) + ', ', end="")
        for name in os.listdir(directory + str(i)):
            arr[j].append(imageio.imread(directory + str(i) + '\\' + name))
        j += 1

    print('')

    data = []
    labels = [] 
    
    print("Reshaping data #: ", end='')
    for c in range(count):
        print(str(i) + ', ', end='')
        for i in range(0, len(arr[c]) - (len(arr[c]) % bag_size), bag_size):
            data.append(arr[c][i:i + bag_size])
            labels.append(proportions[c])
    
    print('')
    
    # to do: shuffle data
    #np.random.shuffle(data)
    #np.random.shuffle(labels)

    return np.array(data), np.array(labels)