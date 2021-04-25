from Simulator import *
from Forward import *
from load_model import *
import numpy as np
from sklearn.metrics import mean_squared_error

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

def rmse(I,computed_I):

    return mean_squared_error(I, computed_I, squared = False)

def mse(I, computed_I):

    return mean_squared_error(I, computed_I, squared = True)

def maximum_entropy(D, computed_I, l = 1):

    chi2 = mse(D, R.T @ computed_I)  

    S = 0 
    p = computed_I/np.sum(computed_I)
    S = -np.sum(p*np.log(p))

    Q = S - l*chi2
    
    return Q



if __name__ == '__main__':

    s = Star(np.pi/4, 5, 3e6, 1e3, 1500)
    s1 = Star(np.pi/4, 5, 3e6, 1e3, 1500)
    I_file_path = './I/I_vector.csv'
    R_parent_directory = './R'
    D_parent_directory = './D'

    R = load_R(R_parent_directory)
    D, wavelengths = load_D(D_parent_directory)
    I = load_I(I_file_path)
    for i in range(len(R)):
        computed_stellar_disk = pseudo_inverse(R[i],D[i])
        true_stellar_disk = s.stellar_disk_vector
        
        true_stellar_disk = s.get_stellar_disk()
        s.plot_on_sphere(computed_stellar_disk, savefig = True, parent_directory='./stellar_disk_computed/')
        s1.plot_on_sphere(true_stellar_disk, savefig = True, parent_directory = './stellar_disk_ground_truths/')

        s.rotate(np.pi * 2 / len(R))
        s1.rotate(np.pi * 2 / len(R))

    #print(rmse(true_stellar_disk, computed_stellar_disk))






   #truth_i45_t5_r3e6_v1e_z1465_wl400_nrot10.png

    
