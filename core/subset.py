import numpy as np
import pandas as pd
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Read the dataset
data = pd.read_csv('../data.csv')
data.drop('Index', axis=1, inplace=True)

# Split the variables from the label
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Select only desired features 
mesh =  mp_face_mesh.FACEMESH_LEFT_EYE |\
        mp_face_mesh.FACEMESH_LEFT_EYEBROW |\
        mp_face_mesh.FACEMESH_LIPS |\
        mp_face_mesh.FACEMESH_RIGHT_EYE |\
        mp_face_mesh.FACEMESH_RIGHT_EYEBROW

features = []

for i in mesh:
    features.append(i[0])
    features.append(i[1])


# Generate a dataframe with only selected features
features = list(set(features))
subset = pd.DataFrame([])

for i in features:
    subset[f'{i}'] = x[f'{i}']

subset['labels'] = y
subset.to_csv('../reduced.csv')