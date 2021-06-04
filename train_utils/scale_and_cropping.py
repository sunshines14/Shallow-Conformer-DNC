import random
import numpy as np


def scale(x):
    new_x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return new_x

def random_cropping(x, cropping_length): 
    start_loc = np.random.randint(0, x.shape[1]-cropping_length)
    new_x = x[:,start_loc:start_loc+cropping_length,:]   
    return new_x