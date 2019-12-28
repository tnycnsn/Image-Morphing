import numpy as np
import dlib
import cv2
from moviepy.editor import *


def edge_points_inserter(subdiv, shape):

    subdiv.insert((0, 0))                           #insert top left corner
    subdiv.insert((0, int(shape[1]/2)))             #insert top edge middle point
    subdiv.insert((0, shape[1]-1))                  #insert top right corner
    subdiv.insert((int(shape[0]/2), shape[1]-1))    #insert right edge middle point
    subdiv.insert((shape[0]-1, shape[1]-1))         #insert bottom right corner
    subdiv.insert((shape[0]-1, int(shape[1]/2)))    #insert bottom edge middle point
    subdiv.insert((shape[0]-1, 0))                  #insert bottom left corner
    subdiv.insert((int(shape[0]/2), 0))             #insert left edge middle point


def construct_triangles(triangles1, points1, points2):

    triangles2 = np.zeros(triangles1.shape)
    for i, tri in enumerate(triangles1):
        Triangle = np.array([tri[::2], tri[1::2]]).T
        target_tri = np.zeros((3,2))
        for j, corner in enumerate(Triangle):

            index = (points1 == corner)
            point_id = np.logical_and(index[:,0], index[:,1])  #find point's id
            if points1[point_id].size == 0: #corner is not in points1 list so it's a edge point
                target_tri[j] = corner
            else:
                corresponded_point = points2[point_id]
                corresponded_point.resize((2,))
                target_tri[j] = corresponded_point

        triangles2[i] = target_tri.flatten("C")

    return triangles2

def make_homogeneous(triangle):
    #Even indices --> x values
    #Odd indices --> y values
    homogeneous = np.array([triangle[::2],  \
                            triangle[1::2], \
                            [1, 1, 1]])
    return homogeneous


def calc_transform(triangle1, triangle2):
    source = make_homogeneous(triangle1).T
    target = triangle2
    Mtx = np.array([np.concatenate((source[0], np.zeros(3))), \
                    np.concatenate((np.zeros(3), source[0])), \
                    np.concatenate((source[1], np.zeros(3))), \
                    np.concatenate((np.zeros(3), source[1])), \
                    np.concatenate((source[2], np.zeros(3))), \
                    np.concatenate((np.zeros(3), source[2]))])

    coefs = np.matmul(np.linalg.pinv(Mtx), target) #get a11, a12, a13, a21, a22, a23
    Transform = np.array([coefs[:3], coefs[3:], [0, 0, 1]])

    return Transform


def vectorised_Bilinear(coordinates, target_img, size):
    coordinates[0] = np.clip(coordinates[0], 0, size[0]-1)
    coordinates[1] = np.clip(coordinates[1], 0, size[1]-1)
    #in order to eliminate overshoot caused by little errors i have clipped it.
    lower = np.floor(coordinates).astype(np.uint32)
    upper = np.ceil(coordinates).astype(np.uint32)

    error = coordinates - lower    #errors = [a, b]
    resindual = 1 - error #resinduals = [1-a, 1-b]

    top_left = np.multiply(np.multiply(resindual[0], resindual[1]).reshape(coordinates.shape[1],1), target_img[lower[0], lower[1], :])
    top_right = np.multiply(np.multiply(resindual[0], error[1]).reshape(coordinates.shape[1],1), target_img[lower[0], upper[1], :])
    bot_left = np.multiply(np.multiply(error[0], resindual[1]).reshape(coordinates.shape[1],1), target_img[upper[0], lower[1], :])
    bot_right = np.multiply(np.multiply(error[0], error[1]).reshape(coordinates.shape[1],1), target_img[upper[0], upper[1], :])

    return np.uint8(np.round(top_left + top_right + bot_left + bot_right))  #Bilinear interpolation for each transformed point


def image_morph(image1, image2, triangles1, triangles2, transforms, t):
    #image morph via backward mapping
    inter_image_1 = np.zeros(image1.shape).astype(np.uint8)
    inter_image_2 = np.zeros(image2.shape).astype(np.uint8)
    result = np.zeros(image2.shape)#.astype(np.uint8)

    for i in range(len(transforms)):

        homo_inter_tri = (1 - t)*make_homogeneous(triangles1[i]) + t*make_homogeneous(triangles2[i])

        polygon_mask = np.zeros(image1.shape[:2], dtype=np.uint8)   #make a given triangular shaped mask
        cv2.fillPoly(polygon_mask, [np.int32(np.round(homo_inter_tri[1::-1, :].T))], color=255)  #Fill inside the polygon to create a mask

        seg = np.where(polygon_mask == 255)     #gives x and y coordinates inside the polygon

        mask_points = np.vstack((seg[0], seg[1], np.ones(len(seg[0]))))   #makes column vectors of [x_i, y_i, 1] those obtained points
        #above 2 lines give us the points which will be transformed

        inter_tri = homo_inter_tri[:2].flatten(order="F")    #make it x1 y1 x2 y2 x3 y3 again in order to pass it to the calc_transform function

        inter_to_img1 = calc_transform(inter_tri, triangles1[i]) #Inverse transform from image1 to inter_tri
        inter_to_img2 = calc_transform(inter_tri, triangles2[i]) #Inverse transform from image2 to inter_tri

        mapped_to_img1 = np.matmul(inter_to_img1, mask_points)[:-1]  #float transfered points without ones  (column vectors)
        mapped_to_img2 = np.matmul(inter_to_img2, mask_points)[:-1]  #float transfered points without ones  (column vectors)

        inter_image_1[seg[0], seg[1], :] = vectorised_Bilinear(mapped_to_img1, image1, inter_image_1.shape)
        inter_image_2[seg[0], seg[1], :] = vectorised_Bilinear(mapped_to_img2, image2, inter_image_2.shape)

    result = (1 - t)*inter_image_1 + t*inter_image_2
    return result.astype(np.uint8)


detector = dlib.get_frontal_face_detector()
#The detector object is used to detect the faces given in an image.
#It works generally better than OpenCV’s default face detector.

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#To predict the landmark points given a face image, a shape predictor with a ready−to−use model is created.
#The model can be found under ”http://dlib.net/files/shapepredictor68facelandmarks.dat.bz2”

image = cv2.imread("selfies/wanted.jpg")

#dim = (int(0.8*image.shape[1]), int(0.8*image.shape[0]))        #Resize dimentions by given ratio
dim = (500, 650)    #Resize dimentions by almost conserving the x/y ratio
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)  #Resize the selfies

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#The predictor only works on grayscale images.
rectangles = detector(gray)
#Use detector to find a list of rectangles containing the faces in the image.
#The rectangles are represented by their xy coordinates.

#Check whether there is only one rectangle. Then, draw the rectangle on the image.
#To reach the values of the rectangle you can use functions given in the official documentation (http://dlib.net/python/index.html#dlib.rectangle).
#E.g. rectangles[0].bl_corner().x and rectangles[0].bl_corner().y will give one of the points. Show your work.
points = predictor(gray, rectangles[0])
#Points is a special structure which stores all the 68 points. By using points.part(i),
#we can reach ith landmark point. You should mark every point on the image. Show your work.

############################ Draw triangles on the Image #########################

img_subdiv = cv2.Subdiv2D((0, 0, image.shape[0], image.shape[1]))
#Subdiv2D is and OpenCV object which performs Delaunay triangulation

search_list = np.zeros((68,2))
for i in range(68):
    img_subdiv.insert((points.part(i).y, points.part(i).x))
    #Each landmark point should be insterted into Subdiv2D object as a tuple
    search_list[i] = [points.part(i).y, points.part(i).x]   #In order to construct triangles in the second image i created search_list

edge_points_inserter(img_subdiv, image.shape)   #Insert 8 of edge points to the subdiv object

img_triangles = img_subdiv.getTriangleList()
#Using get TriangleList function we can obtain the full list of triangles.

############################### Cat Image Processes ############################

cat_img = cv2.imread("cats/00000023_020.jpg")    #Read cat image
x_ratio = 650 / cat_img.shape[0]    #Saved to adjust cat point data
y_ratio = 500 / cat_img.shape[1]    #Saved to adjust cat point

dim = (500, 650)    #Resize dimentions by conserving the x/y ratio
cat_img = cv2.resize(cat_img, dim, interpolation = cv2.INTER_AREA)  #Resize the cat image

cat_info = open("cats/00000023_020.jpg.cat", "r")    #import cat points
str = cat_info.read()
input_list = np.array([int(i) for i in str.split(" ")[1:-1]])   #only the beneficial part is extracted
cat_points = np.reshape(input_list, (-1, 2))    #match x(odd) and y(even) values of given points

ratio_mtx = np.array([[x_ratio, 0], [0, y_ratio]])              #Adjust cat points to ensure compatibility with dim = (500, 650)
cat_points = np.matmul(cat_points, ratio_mtx).astype(np.int)    #Adjust cat points to ensure compatibility with dim = (500, 650)

######################### Process on template points ###########################

template = np.load("template_points.npy")
left_eye = template[36] + np.int32((template[39] - template[36])/2)
right_eye = template[45] + np.int32((template[42] - template[45])/2)

temp_horizontal = np.linalg.norm(right_eye - left_eye)  #distance between eye centers for template
cat_horizontal = np.linalg.norm(cat_points[1] - cat_points[0])  #distance between eye centers for cat

temp_mid = left_eye + np.int32((right_eye - left_eye)/2)
cat_mid = cat_points[0] + np.int32((cat_points[1] - cat_points[0])/2)

temp_vertical = np.linalg.norm(temp_mid - template[66]) #distance between middle point of eyes and mouth center for template
cat_vertical = np.linalg.norm(cat_mid - cat_points[2])  #distance between middle point of eyes and mouth center for cat

horizontal_scale = cat_horizontal / temp_horizontal
vertical_scale = cat_vertical / temp_vertical

scale_mtx = np.array([[vertical_scale, 0.],[0., horizontal_scale]])         #Since dim=0 vertical, dim=1 horizontal axis
template = np.int32(np.round(np.matmul(template, scale_mtx)))               #Scale the template points
trans_vec = np.array([cat_points[2][1], cat_points[2][0]]) - template[66]   #Calculation of the trasnlation vector
template += trans_vec   #translate the template points

########################## Draw triangles on the Cat image ##########################

cat_triangles = construct_triangles(img_triangles, search_list, template)

###################################### Part 5  ####################################

Transforms = np.zeros((len(img_triangles), 3, 3)) #transform matrixes will be saved here
for i in range(len(img_triangles)):
    source = img_triangles[i]          #get source triange
    target = cat_triangles[i]   #get correspondance of the source triangle
    Transforms[i] = calc_transform(source, target)

#################################### OBSERVE ###################################

morphs = []
for t in np.arange(0, 1.0001, 0.02):
    print("processing:\t", t*100, "%")
    morphs.append(image_morph(image, cat_img, img_triangles, cat_triangles, Transforms, t)[:, :, ::-1])

clip = ImageSequenceClip(morphs, fps=25)
clip.write_videofile('part5.mp4', codec='mpeg4')

#cv2.imwrite("cat_first_10_match.jpg", cat_img)
#cv2.imwrite("smile_first_10_match.jpg", image)
#cv2.waitKey()
