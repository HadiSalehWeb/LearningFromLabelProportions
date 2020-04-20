import matplotlib.pyplot as plt
import imageio
import numpy as np
import os
import os.path

def get_cell_data(image_count):
    dir = "D:\Fermentation_2\Tag "
    arr = [[], [], [], [], [], [], []]
    j = 0
    print("Loading cell data #: ", end="")
    for i in [0, 1, 2, 3, 4, 5, 6]:
        print('')
        print(str(i),end="",sep=", ")
        for name in os.listdir(dir + str(i)):
            arr[j].append(imageio.imread(dir + str(i) + '\\' + name))
        j += 1
        
    all_labels = [0.876, 0.872, 0.862, 0.889, 0.846, 0.623, 0.4]

    data = []
    labels = []
    
    print("Reshaping cell data #: ", end='')
    for c in range(len(arr)):
        print(i, end='', sep=', ')
        for i in range(0, len(arr[c]) - (len(arr[c]) % image_count), image_count):
            data.append(arr[c][i:i + image_count])
            labels.append(all_labels[c])
            
    np.random.shuffle(data)
    np.random.shuffle(labels)

    return np.array(data), np.array(labels)