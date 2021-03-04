import cv2 
import numpy as np 

#Creating images
image_vertical  = np.array([[ 0, 255, 0], [ 0, 255, 0] ,[ 0, 255, 0]], np.uint8)
cv2.imwrite('vertical.jpg', image_vertical)
image_horizontal  = np.array([[ 0, 0, 0], [ 255, 255, 255] ,[ 0, 0, 0]], np.uint8)
cv2.imwrite('horizontal.jpg', image_horizontal)


#test
image_vertical_1  = np.array([[ 0, 0, 255], [ 0, 0, 255] ,[ 0, 0, 255]], np.uint8)
image_horizontal_1  = np.array([[ 0, 0, 0], [ 0, 0, 0] , [ 255, 255, 255] ], np.uint8)

cv2.imwrite('test1.jpg', image_vertical_1)
cv2.imwrite('test2.jpg', image_horizontal_1)