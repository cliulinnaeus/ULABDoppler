import numpy as np
import scipy as sp
import math

class Star:


    def __init__(self, inclination_angle, temp, radius, v_e):
        self.inclination_angle = inclination_angle
        self.temp = temp
        self.radius = radius
        self.v_e = v_e
        self.I = None
        self.num_latitudes = 0
        self.zones = None

    def add_sunspots(self, spots_lat, spots_long, spots_radius, spots_temp):
        num_latitudes = self.num_latitudes
        zones = self.zones

        angles = np.zeros(num_latitudes)
        total_polar_angle = np.pi - (np.pi/2 - self.inclination_angle)
        delta_angle = total_polar_angle / num_latitudes 

        #define the latitudes we have in terms of angles
        for i in range(1, len(num_latitudes)+1):
            polar_angle = delta_angle*i
            angles[i] = np.pi/2 - polar_angle
            


        for i in range(len(spots_lat)):
            init_latitude = spots_lat[i]
            init_longitude = spots_long[i]

            final_latitude = 0
            final_longitude = 0

            for j in range(len(angles)):
                if angles[i] > init_latitude:


            



            
        


        


 
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
        num_latitudes, zones = get_lats_and_zones(self,n)
        num_zones = np.sum(zones)
        self.I = np.full( num_zones , self.temp) 
        self.add_sunspots(spots_lat, spots_long, spots_radius, spots_temp)
        



s = Star(0,2,3,4)

print(s.get_lats_and_zones(2000))

a, b = s.get_lats_and_zones(2000)

print(np.sum(b))