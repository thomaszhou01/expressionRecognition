from asyncore import read
import cv2
import numpy as np
from numpy import asarray
from numpy import savetxt
import mediapipe as mp
import pandas as pd

def captureExpression(data, entry):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0)
    count = 0
    startCapture = False


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
                if(count < entry):
                    try:
                        createData(data, results.multi_face_landmarks[0], count)
                        print(count)
                        count += 1
                    except:
                        print("no face")
                else:
                    print("program finished")
                    break
    print(data)
    cap.release()

def createData(data, landmark, num):
    for i in range(0, 468):
        data[num][i][0] = landmark.landmark[i].x
        data[num][i][1] = landmark.landmark[i].y
        data[num][i][2] = landmark.landmark[i].z

def setExpression():
    return input("Expression: 0 = happy, 1 = sad, 2 = angry, 3 = excited, 4 = other: ")
    
while(True):
    num = 400
    faceMeshData = np.zeros(shape = (num,468,3))
    expression = setExpression()
    expression = np.full(num, expression)
    print(expression)
    captureExpression(faceMeshData, num)
    faceMeshData = np.reshape(faceMeshData, (num, 468*3))

    with open('data/labels.csv', 'a') as labelsFile:
        np.savetxt(labelsFile, expression, delimiter=",", fmt = '%s')
    with open('data/data.csv','a') as dataFile:
        np.savetxt(dataFile, faceMeshData, delimiter=",", fmt = '%s')
    
    if(input("continue (0=yes, 1=no)")==1):
        break
