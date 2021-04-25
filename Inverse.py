from Simulator import *
from Forward import *
from load_model import *
import numpy as np
import sklearn  

#TODO: 
# add gaussian noise
# define accuracy measure
# make doppler shift more visible 
# implement maximum entropy method 

def R_transpose(R, D):

    return R.T @ D

def pseudo_inverse(R, D):

    print(np.linalg.pinv(R).shape)
    print(D.shape)

    return np.linalg.pinv(R).T @ D

def accuracy(I,computed_I):

    return sklearn.metrics.mean_squared_error(I, computed_I, squared = False)

if __name__ == '__main__':

    s = Star(np.pi/4, 5, 3e6, 1e3, 1500)

    I_file_path = './I/I_vector.csv'
    R_parent_directory = './R'
    D_parent_directory = './D'

    R = load_R(R_parent_directory)
    D, wavelengths = load_D(D_parent_directory)
    I = load_I(I_file_path)

    pseudo_inverse(R,D)
    s.plot_on_sphere(pseudo_inverse(R, D))
    s.plot_on_sphere(I)
    
