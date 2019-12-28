import numpy as np
import dlib
import cv2

def edge_points_inserter(subdiv, shape):
    subdiv.insert((0, 0))                           #insert top left corner
    subdiv.insert((0, int(shape[1]/2)))             #insert top edge middle point
    subdiv.insert((0, shape[1]-1))                  #insert top right corner
    subdiv.insert((int(shape[0]/2), shape[1]-1))    #insert right edge middle point
    subdiv.insert((shape[0]-1, shape[1]-1))         #insert bottom right corner
    subdiv.insert((shape[0]-1, int(shape[1]/2)))    #insert bottom edge middle point
    subdiv.insert((shape[0]-1, 0))                  #insert bottom left corner
    subdiv.insert((int(shape[0]/2), 0))             #insert left edge middle point


detector = dlib.get_frontal_face_detector()
#The detector object is used to detect the faces given in an image.
#It works generally better than OpenCV’s default face detector.

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#To predict the landmark points given a face image, a shape predictor with a ready−to−use model is created.
#The model can be found under ”http://dlib.net/files/shapepredictor68facelandmarks.dat.bz2”

image = cv2.imread("selfies/smile.jpg")

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

for i in range(68):
    img_subdiv.insert((points.part(i).y, points.part(i).x))
    #Each landmark point should be insterted into Subdiv2D object as a tuple

edge_points_inserter(img_subdiv, image.shape)   #Insert 8 of edge points to the subdiv object

img_triangles = img_subdiv.getTriangleList()
#Using get TriangleList function we can obtain the full list of triangles.

for i in range(len(img_triangles)):
    sel_triangle = img_triangles[i].astype(np.int)
    #You can use cv2.line function to draw the triangles on the image. Show your work here.
    p1 = (sel_triangle[1],sel_triangle[0])  #point1
    p2 = (sel_triangle[3],sel_triangle[2])  #point2
    p3 = (sel_triangle[5],sel_triangle[4])  #point3
    cv2.line(image, p1, p2, [0, 255, 0], thickness=1, lineType=8, shift=0)  #line between p1 and p2
    cv2.line(image, p2, p3, [0, 255, 0], thickness=1, lineType=8, shift=0)  #line between p2 and p3
    cv2.line(image, p3, p1, [0, 255, 0], thickness=1, lineType=8, shift=0)  ##line between p3 and p1

############################### Cat Image Processes ############################

cat_img = cv2.imread("cats/00000023_020.jpg")    #Read cat image

x_ratio = 650 / cat_img.shape[0]    #Saved to adjust cat point data
y_ratio = 500 / cat_img.shape[1]    #Saved to adjust cat point

dim = (500, 650)    #Resize dimentions by conserving the x/y ratio
cat_img = cv2.resize(cat_img, dim, interpolation = cv2.INTER_AREA)  #Resize the cat image

cat_info = open("cats/00000023_020.jpg.cat", "r")    #import cat points
str = cat_info.read()
point_list = np.array([int(i) for i in str.split(" ")[1:-1]])   #only the beneficial part is extracted
cat_points = np.reshape(point_list, (-1, 2))    #match x(odd) and y(even) values of given points

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

cat_subdiv = cv2.Subdiv2D((0, 0, cat_img.shape[0], cat_img.shape[1]))
#Subdiv2D is and OpenCV object which performs Delaunay triangulation

for i in range(68):
    cat_subdiv.insert((template[i,0], template[i,1]))
    #Each landmark point should be insterted into Subdiv2D object as a tuple

edge_points_inserter(cat_subdiv, cat_img.shape)   #Insert 8 of edge points to the subdiv object

cat_triangles = cat_subdiv.getTriangleList()
#Using get TriangleList function we can obtain the full list of triangles.

for i in range(len(cat_triangles)):
    sel_triangle = cat_triangles[i].astype(np.int)
    #You can use cv2.line function to draw the triangles on the image. Show your work here.
    p1 = (sel_triangle[1],sel_triangle[0])  #point1
    p2 = (sel_triangle[3],sel_triangle[2])  #point2
    p3 = (sel_triangle[5],sel_triangle[4])  #point3
    cv2.line(cat_img, p1, p2, [0, 255, 0], thickness=1, lineType=8, shift=0)  #line between p1 and p2
    cv2.line(cat_img, p2, p3, [0, 255, 0], thickness=1, lineType=8, shift=0)  #line between p2 and p3
    cv2.line(cat_img, p3, p1, [0, 255, 0], thickness=1, lineType=8, shift=0)  ##line between p3 and p1

#################################### OBSERVE ###################################

cv2.imshow("cat", cat_img)
cv2.imshow("image", image)
cv2.waitKey()

#cv2.imwrite("Report/part3_smile.jpg", image)
#cv2.imwrite("Report/part3_cat.jpg", cat_img)
