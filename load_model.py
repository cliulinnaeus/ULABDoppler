import numpy as np
import pandas as pd
import glob, os


def load_R(parent_directory):

    array_lst = []
    for file in glob.glob(parent_directory + "/*.csv"):
        array = np.loadtxt(file, delimiter = ',')
        array_lst.append(array)

    return np.hstack(tuple(array_lst))
    

def load_I(file_path):
    
    df = pd.read_csv(file_path)
    
    return np.array(df['brightness']) 


def load_D(parent_directory):
    
    array_lst = []
    global wavelengths
    for file in glob.glob(parent_directory + "/*.csv"):
        df = pd.read_csv(file)
        wavelengths = np.array(df['wavelength'])
        flux = np.array(df['flux'])

        array_lst.append(flux)

    return np.concatenate(tuple(array_lst)), wavelengths
        
    

if __name__ == '__main__': 

    I_file_path = './I/I_vector.csv'
    R_parent_directory = './R'
    D_parent_directory = './D'

    R = load_R(R_parent_directory)
    D, wavelengths = load_D(D_parent_directory)
    I = load_I(I_file_path)

    print(R.shape)
    print(D.shape)
    print(wavelengths.shape)
    print(I.shape)
    



