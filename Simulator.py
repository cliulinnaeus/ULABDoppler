import numpy as np
import scipy as sp
import math
import copy
import matplotlib.pyplot as plt


class Star:

    def __init__(self, inclination_angle, temp, radius, v_e, num_of_patches):
        spots_lat = np.array([0, np.pi/2])
        spots_long = np.array([np.pi, np.pi/2])
        spots_radius = np.array([10, 10])
        spots_temp = np.array([3000, 1000])



        self.inclination_angle = inclination_angle
        self.temp = temp
        self.radius = radius
        self.v_e = v_e
        self.num_latitudes, self.zones = self.get_lats_and_zones(num_of_patches)
        self.I = self.make_image_vector(num_of_patches, spots_lat, spots_long, spots_radius, spots_temp)


    def add_sunspots(self, I, spots_lat, spots_long, spots_radius, spots_temp):
        num_latitudes = int(self.num_latitudes)
        zones = self.zones

        angles = np.zeros(num_latitudes)
        total_polar_angle = np.pi - (np.pi/2 - self.inclination_angle)
        delta_angle = total_polar_angle / num_latitudes

        #define the latitudes we have in terms of angles
        for i in range(1, num_latitudes+1):
            polar_angle = delta_angle*i
            angles[i-1] = np.pi/2 - polar_angle

        for x in range(len(spots_lat)):
            init_latitude = spots_lat[x]
            init_longitude = spots_long[x]

            i = math.floor((np.pi/2 - init_latitude) / delta_angle)

            delta_phi = 2*np.pi / zones[i]

            j = math.floor(init_longitude / delta_phi)

            index = int(np.sum(self.zones[0:i])) + j
            I[index] = spots_temp[x]
        return I

    def get_lats_and_zones(self, n):

        area_per_patch = self.calc_zone_area(self.radius, self.inclination_angle, n)
        length_zone = np.sqrt(area_per_patch)

        #calculate how many "latitudes" on sphere
        total_polar_angle = np.pi - (np.pi/2 - self.inclination_angle)
        num_latitudes = int(math.floor((total_polar_angle * self.radius) / length_zone))

        #calculate how many zones for each latitude
        delta_angle = total_polar_angle / num_latitudes

        zones = np.zeros(num_latitudes)

        for i in range(1, len(zones)+1):
            polar_angle = delta_angle*i
            num_zones = (2 * np.pi * self.radius * np.sin(polar_angle)) / length_zone

            zones[i-1] = int(math.floor(num_zones))

        self.num_latitudes = num_latitudes
        self.zones = zones

        return num_latitudes, zones


    def get_stellar_disk(self, phase):

        return None

    def calc_zone_area(self, radius, inclination_angle, n):

        total_area = (4 * np.pi * radius**2) - (2 * np.pi * radius**2)*(1 - np.sin(inclination_angle))
        area_per_patch = total_area/n

        return area_per_patch


    def make_image_vector(self, n, spots_lat, spots_long, spots_radius, spots_temp):
        num_latitudes = self.num_latitudes
        num_zones = int(np.sum(self.zones))
        I = np.full(num_zones, self.temp)
        I = self.add_sunspots(I, spots_lat, spots_long, spots_radius, spots_temp)
        return I

    def _sort_into_bins(self):
        arr = []
        start_idx = 0
        for z in self.zones:
            z = int(z)
            arr.append(copy.deepcopy(self.I[start_idx: start_idx + z]))
            start_idx = z + start_idx
        return arr

    def plot_on_sphere(self):
        I = self.I
        counter = 0
        n = len(I)
        inclination_angle = self.inclination_angle
        num_latitudes = self.num_latitudes
        zones = self.zones
        width = int(max(zones))
        height = int(num_latitudes)
        map = np.zeros((height, width))
        bins = self._sort_into_bins()
        for idx, bin in enumerate(bins):
            start_col = (width - len(bin)) // 2
            map[idx][start_col : start_col + len(bin)] = bin
        plt.imshow(map, cmap='hot')
        plt.show()



s = Star(np.pi/2, 5000, 3e6, 4, 10000)
# s.make_image_vector(2000, np.array([0]), np.array([0]),1,np.array([2000]))


s.plot_on_sphere()

# print(s.I)