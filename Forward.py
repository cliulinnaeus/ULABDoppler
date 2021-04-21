import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import constants
from scipy.integrate import quad
from Simulator import * 

h = constants.Planck
c = constants.c
k = constants.k
inf = np.inf

def black_body(wavelength, temperature):
    """
    Input: wavelength, temperature
    Output: intensity calculated by B(l, T)
    """
    l, T = wavelength, temperature
    return 2*h*(c**2)/(l**5)*1/((np.exp(h*c/(l*k*T)))-1)

def integrate_black_body(init_wavelength, delta_wavelength, temperature):

    return quad(black_body, init_wavelength, init_wavelength+delta_wavelength, args=(temperature))[0]


def get_temperature(I):
    """
    Input: Image vector I
    Output: Temperature of each element(i) in I, T(i); type = 1D_array
    """
    return I


def black_body_matrix(temp, frac=[1, 1]):
    """
    Input: wavelength, temperature; type: np.array
    Output: matrix constructed using black body radiation
    """
    B = []
    for f in frac:
        Bi = []
        for T in temp:
            B_ij = np.array([quad(black_body, 0, inf, args=(T))[0]])
            Bi.append(B_ij*f)
        B.extend(Bi)
    B_mat = np.array(B)
    B_mat.reshape(len(temp), len(frac))
    return np.array(B_mat)

def doppler_shift(v_rot):
    """
    Input: rotational velocity of the star
    Output: the doppler shift fraction of wavelength using equation $v_rot/c = lambda / lambda_0$
    """
    return v_rot / c + 1

print(black_body_matrix([1]))

def get_R(star, num_wavelengths):
    
    max_wavelength = 5000e-9
    delta_wavelength = max_wavelength / num_wavelengths #meters

    star.get_stellar_disk()
    stellar_disk_vector = star.stellar_disk_vector

    num_latitudes = star.num_latitudes
    inclination_angle = star.inclination_angle
    zones = star.zones 
    
    wavelength_lst = np.linspace(0, max_wavelength, num_wavelengths)
    temp_lst = stellar_disk_vector**(1/4) / sigma

    R = [] 

    for i in range(len(stellar_disk_vector)):
        row = []
        
        if stellar_disk_vector[i] != 0.0:
            for j in range(num_wavelengths):

                a = integrate_black_body(wavelength_lst[j], delta_wavelength, temp_lst[i])
                normalized_flux = a / stellar_disk_vector[i]
                row.append(normalized_flux)

        else: 
            for j in range(num_wavelengths):
                row.append(0.0)
                
        R.append(row)
    
    R = np.array(R)
    
    return R

s = Star(np.pi/2, 5000, 3e6, 4, 10000)

stellar_disk = s.get_stellar_disk()

R = get_R(s, 400)

line_spectra = R.T @ stellar_disk

wavelengths = np.linspace(0,5000e-9,400)

plt.plot(wavelengths ,line_spectra)
plt.show()
    


    




