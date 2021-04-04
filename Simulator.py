import numpy as np
import scipy as sp
import math

class Star:


    def __init__(self, inclination_angle, temp, radius, v_e):
        self.inclination_angle = inclination_angle
        self.temp = temp
        self.radius = radius
        self.v_e = v_e
        self.I = 0

    def add_sunspots(self, I, spots_lat, spots_long, spots_radius, spots_temp):
        
        
 
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
        
        return num_latitudes, zones          


    def get_stellar_disk(self, phase):

        return None

    def calc_zone_area(self, radius, inclination_angle, n):

        total_area = (4 * np.pi * radius**2) - (2 * np.pi * radius**2)*(1 - np.sin(inclination_angle))
        area_per_patch = total_area/n
        
        return area_per_patch

    def add_background(I, temp):
        sigma = 5.67e-8
        flux = sigma * temp**4
        
        for i in range(len(I)):
            I[i] = temp

        return I



    def make_image_vector(self, n, spots_lat, spots_long, spots_radius, spots_temp):
        num_latitudes, zones = get_lats_and_zones(self,n)
        num_zones = np.sum(zones)
        I = np.full( num_zones , self.temp) 
        #I = self.add_background(I, self.temp)
        I = self.add_sunspots(I, spots_lat, spots_long, spots_radius, spots_temp)
        self.I = I



s = Star(0,2,3,4)

print(s.get_lats_and_zones(2000))

a, b = s.get_lats_and_zones(2000)

print(np.sum(b))