import numpy as np
import scipy as sp

class Star:


    def __init__(self, inclination_angle, temp, radius, v_e):
        self.inclination_angle = inclination_angle
        self.temp = temp
        self.radius = radius
        self.v_e = v_e
        self.I = 0

    def add_sunspots(self, I, latitude, longitude, radius, temp):

        return None


    def get_stellar_disk(self, phase):

        return None