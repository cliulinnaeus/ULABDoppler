import numpy as np
import matplotlib.pyplot as plt
import glob


# read each csv to
file_name = "./HD199178_spectra/J_A+A_625_A79_sp_HD199178_1994_07-49552.340.dat.csv"

def make_file_name(root_dir, year, month, HJD):
    baseline = "/J_A+A_625_A79_sp_HD199178_"
    file_name = baseline + str(year) + "_" + f"{month:02}" + "-" + "{0:.3f}".format(HJD)
    file_name = file_name + ".dat.csv"
    return root_dir + file_name



def read_single_file(file_name):
    wavelength = []
    flux = []
    with open(file_name) as f:
        lis = np.array([line.split() for line in f])
        for data in lis[4 : ]:
            wavelength.append(float(data[0]))
            flux.append(float(data[1]))
    return np.array(wavelength), np.array(flux)


root_dir = "./HD199178_spectra"
all_files = glob.glob(root_dir + "/*.csv")
wavelength_lst, flux_lst = [], []
for file in all_files:
    wavelength, flux = read_single_file(file)
    wavelength_lst.append(wavelength)
    flux_lst.append(flux)

def filtered_coord(wavelength, flux):
    """
    Input: unfilered lists of wavelength and flux; type == 2D array
    Output: new lists of wavelength and flux s.t. new_lst = [[old1], [old2]] if a gap exists between [old1] and
        [old2]; type == 2D array: new_wavelength, new_flux

    Precondition: lists of wavelength and flux have the same size
    """
    assert len(wavelength) == len(flux), 'Coordinates have the same size'

    new_wl = []
    new_fx = []
    i = 0
    while i < len(wavelength):
        indv_wl, indv_fx = [wavelength[i][0]], [flux[i][0]]
        j = 1
        while j < len(wavelength[i]):
            if is_gap(indv_wl, wavelength[i][j]):
                new_wl.append(indv_wl)
                new_fx.append(indv_fx)
                indv_wl, indv_fx = [wavelength[i][j]], [flux[i][j]]
            else:
                indv_wl.append(wavelength[i][j])
                indv_fx.append(flux[i][j])
            j += 1
        new_wl.append(indv_wl)
        new_fx.append(indv_fx)
        i += 1
    return new_wl, new_fx

def filtered_coord_single(wavelength, flux, chunk_num):
    """
    Input: unfilered lists of wavelength and flux; type == 2D array
    Output: new lists of wavelength and flux s.t. new_lst = [[old1], [old2]] if a gap exists between [old1] and
        [old2]; type == 2D array: new_wavelength, new_flux

    Precondition: lists of wavelength and flux have the same size
    """
    assert len(wavelength) == len(flux), 'Coordinates have the same size'

    new_wl = []
    new_fx = []
    i = 1
    indv_wl, indv_fx = [wavelength[0]], [flux[0]]
    while i < len(wavelength):
        if is_gap(indv_wl, wavelength[i]):
            new_wl.append(indv_wl)
            new_fx.append(indv_fx)
            indv_wl, indv_fx = [wavelength[i]], [flux[i]]
        else:
            indv_wl.append(wavelength[i])
            indv_fx.append(flux[i])
        new_wl.append(indv_wl)
        new_fx.append(indv_fx)
        i += 1
    print("Number of chunks: " + str(len(new_wl)))
    return new_wl[chunk_num], new_fx[chunk_num]

def is_gap(cur_lst, item):
    """ 
    Identify if a gap exists between the current list item and the next item based on the difference in mean.
    If the difference in mean between the new next element and current element in wavelength is 10x bigger 
    than the mean between the current element and the previous element, return True; False otherwise.

    Precondition: cur_lst has at least one element
    """
    assert len(cur_lst) > 0, 'empty list'
    
    if len(cur_lst) == 1:
        return False
    last_elem = cur_lst[len(cur_lst)-1]
    sec_last = cur_lst[len(cur_lst)-2]
    if (10 * abs(last_elem - sec_last)) < abs(item - last_elem):
        return True
    return False

new_wavelength, new_flux = filtered_coord(wavelength_lst, flux_lst)

def read_file(year, month, HJD):
    file_name = make_file_name(root_dir, year, month, HJD)
    x, y = read_single_file(file_name)
    def select_chunk(chunk_num):
        return filtered_coord_single(x, y, chunk_num)
    return select_chunk
    


"""
fig = plt.figure()
ax = fig.add_subplot(111)
for w, f in zip(new_wavelength, new_flux):
    ax.clear()
    ax.plot(w, f)
    plt.pause(0.1)
"""
a = read_file(1994, 7, 49562.352)
print(a(0))
#print(new_wavelength)






