from Simulator import *
from Forward import *
from load_model import *
import numpy as np
from scipy.optimize import minimize 
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

def maximum_entropy(computed_I, D, l = 10):

    chi2 = mse(D, R.T @ computed_I)  

    S = 0 
    p = computed_I/np.sum(computed_I)
    S = -np.sum(p*np.log(p))

    Q = S - l*chi2
    
    return -Q





    
    


if __name__ == '__main__':

    s = Star(np.pi/4, 5, 3e6, 1e3, 50)
    s1 = Star(np.pi/4, 5, 3e6, 1e3, 50)
    I_file_path = './I/I_vector.csv'
    R_file_path = './R/R_matrix.csv'
    D_file_path = './D/flux_vs_wavelength_data.csv'

    R = load_R(R_file_path)
    print(R.shape)
    D, wavelengths = load_D(D_file_path)
    I = load_I(I_file_path)
    
    computed_stellar_disk = pseudo_inverse(R,D)
    
    #true_stellar_disk = s.get_stellar_disk()
    s.plot_on_sphere(computed_stellar_disk, savefig = True, parent_directory='./stellar_disk_computed/')
    s1.plot_on_sphere(I, savefig = True, parent_directory = './stellar_disk_ground_truths/')

    print(computed_stellar_disk[len(computed_stellar_disk)-2])
    #print(rmse(true_stellar_disk, computed_stellar_disk))

    s.plot_on_sphere(computed_stellar_disk-I)

    '''optimize'''
    
    I0 = np.full(I.shape, 5, dtype = float)
    
    result = minimize(maximum_entropy, x0 = I0, args = (D,))

    optimized_I = result.x

    s.plot_on_sphere(optimized_I)

    print(result.success)


   #truth_i45_t5_r3e6_v1e_z1465_wl400_nrot10.png

    
#TRY WITH INCLINATION pi/2, different inclination
#OMIT DOPPLER IMAGING
#MAKE D AND R HAVE DIFFERENT BUT CLOSE TEST STARS