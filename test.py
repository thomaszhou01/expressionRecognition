from asyncore import read
import cv2
import numpy as np
from numpy import asarray
from numpy import savetxt
import mediapipe as mp
import pandas as pd
import tensorflow as tf


def captureExpression():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0)
    startCapture = False

    #data
    faceMeshData = np.zeros(shape = (468,3))
    model = tf.keras.models.load_model('models/jan23.h5')

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
            width  = cap.get(3)
            height = cap.get(4)

            cv2.circle(image,(int(width/2),int(height/2)), 150, (0,0,255), 1)

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            if cv2.waitKey(1) & 0xFF == 27:
                break
            elif cv2.waitKey(1) & 0xFF == 32:
                startCapture = True
            
            if startCapture:
                try:
                    createData(faceMeshData, results.multi_face_landmarks[0])
                    x = model.predict(np.reshape(faceMeshData, (-1, 1404)))
                    maxNum = max(x[0])
                    indexNum = np.where(x[0] == maxNum)
                    indexNum = indexNum[0][0]
                    print(indexNum)
                    if(indexNum == 0):
                        print("happy")
                    elif(indexNum == 1):
                        print("sad")
                    elif(indexNum == 2):
                        print("angry")
                    elif(indexNum == 3):
                        print("excited")
                except Exception as e:
                    print(e)
    cap.release()

def createData(data, landmark):
    for i in range(0, 468):
        data[i][0] = landmark.landmark[i].x
        data[i][1] = landmark.landmark[i].y
        data[i][2] = landmark.landmark[i].z

def setExpression():
    return input("Expression: 0 = happy, 1 = sad, 2 = angry, 3 = excited, 4 = other: ")
    

captureExpression()
