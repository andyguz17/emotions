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
        fails = 0
        success = 0
        
        for i, name in enumerate(os.listdir(f'{path}')):
            img = cv2.imread(f'{path}/' + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            data = self.mesh.detect(img)
            if (data == []):
                fails += 1
                continue

            success += 1
            data = self.features.getDistances(data, coord)
            distances.append(np.array(data))

        print(f'{path} \n\t success: {success} \n\t fails: {fails}')
        distances = pd.DataFrame(distances)
        distances['label'] = label

        return distances
