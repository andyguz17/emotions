"""
Use mediapipe extract features from an image

@detect: 
	inputs: 
		- image: the image to analyze as a cv variable 
		
	returns a dictionary with detection coordinates  
"""

import cv2
import mediapipe as mp


class Detector():

    mp_face_mesh = mp.solutions.face_mesh

    def detect(self, image):
        with self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5) as face_mesh:

            image = cv2.resize(
                image, (int(image.shape[1] * 20), int(image.shape[0] * 20)))
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.multi_face_landmarks:
                return []

            return results.multi_face_landmarks
