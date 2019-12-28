import numpy as np
import dlib
import cv2


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

for i in range(9):
    cat_img[cat_points[i,1]-3:cat_points[i,1]+3, cat_points[i,0]-3:cat_points[i,0]+3, :] = [0, 255, 0]
    #make given point(and around) green

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
trans_vec = np.array([cat_points[2][1], cat_points[2][0]]) - template[66]   #Calculation of the translation vector
template += trans_vec   #translate the template points

for i in range(68):
    cat_img[template[i,0]-3:template[i,0]+3, template[i,1]-3:template[i,1]+3, :] = [255, 0, 0]
    #make given points(and around) blue

#################################### OBSERVE ###################################

cv2.imshow("cat", cat_img)
cv2.waitKey()

#cv2.imwrite("Report/part2_cat.jpg", cat_img)
