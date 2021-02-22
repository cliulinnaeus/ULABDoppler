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



# file_name = make_file_name(root_dir, 1994, 7, 49552.340)
#
#
# x, y = read_single_file(file_name)
fig = plt.figure()
ax = fig.add_subplot(111)
for w, f in zip(wavelength_lst, flux_lst):
    ax.clear()
    ax.plot(w, f)
    plt.pause(0.01)






