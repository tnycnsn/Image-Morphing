import numpy as np
import dlib
import cv2
from moviepy.editor import *


source = np.array([[100, 100], [100, 600], [200, 100]])
target = np.array([[600, 300], [500, 600], [400, 100]])

############################# Calculate Transform Matrix ##########################

q = target.flatten()    #make it vectorize

M = np.array([[source[0][0], source[0][1], 1, 0, 0, 0], \
            [0, 0, 0, source[0][0], source[0][1], 1],   \
            [source[1][0], source[1][1], 1, 0, 0, 0],   \
            [0, 0, 0, source[1][0], source[1][1], 1],   \
            [source[2][0], source[2][1], 1, 0, 0, 0],   \
            [0, 0, 0, source[2][0], source[2][1], 1]])

coefs = np.matmul(np.linalg.pinv(M), q) #get a11, a12, a13, a21, a22, a23
Affine_transform = np.array([coefs[0:3], coefs[3:], [0, 0, 1]])

#################################################3##############################

new_source = np.concatenate((source, np.ones((3,1))), axis=1)   #make points homogenous

steps = 20
frame_list = []

for t in range(steps + 1):
    canvas = np.zeros((700, 700, 3))

    current_points = (1 - t/steps)*new_source.T + (t/steps)*np.matmul(Affine_transform, new_source.T)   #calculate translated points for t value

    cv2.polylines(canvas, np.int32([current_points[:-1].T]), isClosed=True, color=(int((t/steps)*255), 0, int((1 - t/steps)*255)), thickness=1)
    cv2.fillPoly(canvas, np.int32([current_points[:-1].T]), color=(int((t/steps)*255), 0, int((1 - t/steps)*255)))

    frame_list.append(canvas[:, :, ::-1])   #make it BGR ro RGB and append


clip = ImageSequenceClip(frame_list, fps=25)
clip.write_videofile('part4.mp4', codec='mpeg4')
