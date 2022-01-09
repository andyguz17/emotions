"""
Helps to work with mediapipe coordinates:

@ coordinates:
    inputs: 
        points: a dictionary of mediapipe points

    returns the coordinates from a point dictionary

@ getDistances: 
    inputs:
        - points: the points to calculate the distance
        - arch: decides if the points are 2 or 3 dimensional

    returns the distance between two points 
    
"""

import numpy as np
import math


class Features():

    def coordinates(self, points):
        x = []
        y = []
        z = []

        for data_point in points:

            for landmark in data_point.landmark:
                x.append(landmark.x)
                y.append(landmark.y)
                z.append(landmark.z)

        x = 1 - np.array(x)
        y = 1 - np.array(y)
        z = 1 - np.array(z)

        return x, y, z

    def getDistances(self, points, arch="2d"):

        x, y, z = self.coordinates(points)

        labels = np.arange(0, 468)
        distances = []

        for i in labels:
            if (arch == "2d"):
                distance = math.sqrt(
                    math.pow(x[4] - x[i], 2) + math.pow(y[4] - y[i], 2))
                distances.append(distance)

            elif (arch == "3d"):
                distance = math.sqrt(
                    math.pow(x[4] - x[i], 2) + math.pow(y[4] - y[i], 2) +
                    math.pow(z[4] - z[i], 2))
                distances.append(distance)

        return np.array(distances)
