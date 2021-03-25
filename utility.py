import numpy as np
import scipy as sp
from file_reader import *


"""
1. function: num of regions ----> a vector where each element is the brightness of a region
2. forward: R, I -----> D
3. inverse: D, R -----> I

"""

def vector_to_map(I, radius, inclination_angle):
    def get_2D_map(I, radius):
        num = len(I)
        num_lat = np.sqrt(np.pi * num) / 2
        b = 2 * radius * np.sqrt(np.pi / num)
        surf_map = []
        lat = 1
        while lat <= num_lat:
            theta = 180 / (np.pi * lat * b)
            num_long = np.sqrt(num * np.pi) * np.sin(theta)
            long = 0
            while long < num_long:
                surf_ring = []
                surf_ring.append(I[long])
                long += 1
            surf_map.append(surf_ring)
            lat += 1
        return surf_map
    
    def plot(surf_map, incl_angle):
        """
        plot the sphere given the 2D array and inclination angle
        """
        return None
    
    return plot(get_2D_map, inclination_angle)


def forward(R, I):
    #TODO:
    return None



def inverse(R, D):
    #TODO:
    return None




