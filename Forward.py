import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import constants
from scipy.integrate import quad
from Simulator import * 

# setting these constants under SI unit will cause 32 bit float overflow
# h = constants.Planck
# c = constants.c
# k = constants.k

# T -> kilo Kelvin
# lambda (wavelength) -> 10e-7 -> 10e-4 m -> in units of anstrom

h = 1
c = 1
k = 1

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


def get_temperature(star):
    """
    Input: star
    Output: Temperature of each element(i) in I, T(i); type = 1D_array
    """
    return star.I


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

def get_v_radial(star, index):
    """
    Input: star, index of the patch in image vector I, converting into: w = angular velocity, theta & phi = angles of each patch, R = radius, i = inclination angle
    Output: radial speed
    """
    R = star.radius
    w = star.v_e / R
    lat, lon = star.get_lat_lon(star.I, index)
    #print(lat, lon)
    theta = np.pi/2 - lat
    phi = lon
    #print(theta, phi)
    i = star.inclination_angle

    v_ang = np.array([0, 0, w])
    position = np.array([R*np.sin(theta)*np.cos(phi), R*np.sin(theta)*np.sin(phi), R*np.cos(theta)])
    inclination = np.array([0, np.sin(i), np.cos(i)])

    v_radial = np.dot(np.cross(v_ang, position), inclination)
    return v_radial

def doppler_shift(star, l0):
    """
    Input: star, wavelength of the source (using Black Body radiation); converting into: rotational velocity of the star
    Output: the doppler shift fraction of wavelength using equation $l/l0 = sqrt((1-B)/(1+B))$
    """
    v_r = np.array([get_v_radial(star, i) for i in range(len(star.I))])

    #l0 = np.array([quad(black_body, 0, inf, args=(T))[0] for T in temperature])
    B = v_r / c
    l = np.sqrt((1-B)/(1+B))*l0
    return l




def get_R(star, num_wavelengths, max_wavelength = 15000):
    
    delta_wavelength = max_wavelength / num_wavelengths #meters

    star.get_stellar_disk()
    stellar_disk_vector = star.stellar_disk_vector
    #TODO: comment this line out and uncomment the above line
    # stellar_disk_vector = star.I

    num_latitudes = star.num_latitudes
    inclination_angle = star.inclination_angle
    zones = star.zones 
    
    wavelength_lst = np.linspace(0, max_wavelength, num_wavelengths)
    temp_lst = np.power(stellar_disk_vector, 0.25) / sigma
    # plt.imshow(temp_lst.reshape((1, len(temp_lst))))
    # plt.show()


    R = []

    for i in range(len(stellar_disk_vector)):
        row = []
        
        if stellar_disk_vector[i] != 0.0:
            for j in range(num_wavelengths):
                a = integrate_black_body(wavelength_lst[j], delta_wavelength, temp_lst[i])
                normalized_flux = a / stellar_disk_vector[i]
                row.append(doppler_shift(star, normalized_flux))

        else: 
            for j in range(num_wavelengths):
                row.append(0.0)
                
        R.append(row)
    R = np.array(R)
    #R = np.array([doppler_shift(star, l0) for l0 in R])
    return R

if __name__ == '__main__':
    s = Star(np.pi/2, 5, 3e6, 4, 500)

    phi_list = list(range(0, 10))
    line_spectra_lst = []
    for _ in phi_list:
        s.rotate(np.pi * 2 / len(phi_list))

        stellar_disk = s.get_stellar_disk()
        max_wavelength = 3
        R = get_R(s, 400, max_wavelength=max_wavelength)
        #s.plot_on_sphere(s.stellar_disk_vector)
        #
        # f = plt.imshow(R)
        # plt.colorbar(f)
        # plt.show()

        # print(R.shape)
        # print(stellar_disk.shape)

        line_spectra = R.T @ stellar_disk
        line_spectra_lst.append(line_spectra)
    s_new = Star(np.pi/2, 5, 3e6, 4e6, 500)
    R = get_R(s_new, 400)
    f = plt.imshow(R)
    plt.colorbar(f)
    plt.show()

'''
    wavelengths = np.linspace(0, max_wavelength, 400)
    for l in line_spectra_lst:
        plt.plot(wavelengths, l)
    plt.legend(phi_list)
    plt.show()

'''



    


    




