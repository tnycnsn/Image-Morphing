import numpy as np
import dlib
import cv2

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

for i in range(68):
    image[points.part(i).y-3:points.part(i).y+3, points.part(i).x-3:points.part(i).x+3, :] = [0, 255, 0]
    #dlib.points.y indicate vertical and dlib.points.x indicate horizontal axis, for image data it's vice versa.

############################### Cat Image Processes ############################

cat_img = cv2.imread("cats/00000023_020.jpg")
x_ratio = 650 / cat_img.shape[0]    #Saved to adjust cat point data
y_ratio = 500 / cat_img.shape[1]    #Saved to adjust cat point

dim = (500, 650)    #Resize dimentions by conserving the x/y ratio
cat_img = cv2.resize(cat_img, dim, interpolation = cv2.INTER_AREA)  #Resize the cat image

cat_info = open("kediler/00000023_020.jpg.cat", "r")
str = cat_info.read()
point_list = np.array([int(i) for i in str.split(" ")[1:-1]])   #only the beneficial part is extracted
cat_points = np.reshape(point_list, (-1, 2))    #match x(odd) and y(even) values of given points

ratio_mtx = np.array([[x_ratio, 0], [0, y_ratio]])              #Adjust cat points to ensure compatibility with dim = (500, 650)
cat_points = np.matmul(cat_points, ratio_mtx).astype(np.int)    #Adjust cat points to ensure compatibility with dim = (500, 650)

for i in range(9):
    cat_img[cat_points[i,1]-5:cat_points[i,1]+5, cat_points[i,0]-5:cat_points[i,0]+5, :] = [0, 255, 0]
    #make given point green

#################################### OBSERVE ###################################

cv2.imshow("image", image)
cv2.imshow("cat image", cat_img)
cv2.waitKey()

#cv2.imwrite("Report/part1_smile.jpg", image)
#cv2.imwrite("Report/part1_cat.jpg", cat_img)
