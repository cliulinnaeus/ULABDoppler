import numpy as np
import scipy as sp
import math
import copy

class Star:


    def __init__(self, inclination_angle, temp, radius, v_e):
        self.inclination_angle = inclination_angle
        self.temp = temp
        self.radius = radius
        self.v_e = v_e
        self.I = None
        self.num_latitudes = 0
        self.zones = None

    def _sort_into_bins(self):
        arr = []
        start_idx = 0
        for z in self.zones:
            arr.append(copy.deepcopy(self.I[start_idx: start_idx + z]))
            start_idx = z
        return arr


    def add_sunspots(self, spots_lat, spots_long, spots_radius, spots_temp):
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

            j = math.floor((init_longitude) / delta_phi)

            index = int(np.sum(self.zones[0:i])) + j

            self.I[index] = spots_temp[x]



            





            



            
        


        


 
        return None

    def get_lats_and_zones(self, n):

        area_per_patch = self.calc_zone_area(self.radius, self.inclination_angle, n)
        length_zone = np.sqrt(area_per_patch)
        
        #calculate how many "latitudes" on sphere
        total_polar_angle = np.pi - (np.pi/2 - self.inclination_angle)
        num_latitudes = math.floor((total_polar_angle * self.radius) / length_zone)

        #calculate how many zones for each latitude
        delta_angle = total_polar_angle / num_latitudes 

        zones = np.zeros(num_latitudes)

        for i in range(1, len(zones)+1):
            polar_angle = delta_angle*i 
            num_zones = (2 * np.pi * self.radius * np.sin(polar_angle)) / length_zone

            zones[i-1] = math.floor(num_zones)

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
        num_latitudes, zones = self.get_lats_and_zones(n)
        num_zones = np.sum(zones)
        self.I = np.full( num_zones , self.temp) 
        self.add_sunspots(spots_lat, spots_long, spots_radius, spots_temp)
        



s = Star(np.pi/2,5000,3,4)

s.make_image_vector(2000, np.array([0]), np.array([0]),1,np.array([2000]))

print(s.I) 