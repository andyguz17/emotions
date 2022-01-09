"""
Use the mesh class to create a dataframe with mediapipe coordinates and label

@setData: 
	inputs: 
		- path: the path to the faces images
		- label: the label of the corresponding image (angry, sad, etc)
        - coord: decides if the coordinates are 2 or 3 dimensional 

	returns a dataframe with the coordinates of the points detected and its label 
"""

import os
import cv2
import mesh
import coordinates
import numpy as np
import pandas as pd


class meshFrame():

    mesh = mesh.Detector()
    features = coordinates.Features()

    def setData(self, path, label, coord="2d"):

        distances = []

        for i, name in enumerate(os.listdir(f'{path}')):
            img = cv2.imread(f'{path}/' + name)
            img = cv2.resize(
                img, (int(img.shape[1] * 0.2), int(img.shape[0] * 0.2)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            data = self.mesh.detect(img)
            if (data == []):
                continue

            data = self.features.getDistances(data, coord)
            distances.append(np.array(data))

        distances = pd.DataFrame(distances)
        distances['label'] = label
        print('done')

        return distances
