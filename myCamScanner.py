# shay finegold 311165609 , Dan monsonego , 313577595

from itertools import count
import numpy as np
import cv2 as cv
import matplotlib.pylab as plt
import glob
import sys
import os
import math


args = sys.argv  # getting args from the command line
input_img = args[1]
output_dir = args[2]


img = cv.imread(input_img, cv.IMREAD_COLOR)  # reading the image
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert bgr to gray
# aplaying binarization that halps in finding the contours
ret, thresh = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)

contours, hierarchy = cv.findContours(
    image=thresh, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)  # finding all the external contours


# findling the largest contour (ractangle)
largestContour = max(contours, key=cv.contourArea)

image_copy = img.copy()  # creating an image copy

cv.drawContours(image=image_copy, contours=[largestContour], contourIdx=0,
                color=(0, 255, 0), thickness=40, lineType=cv.LINE_AA)  # drawing the rec contour on our image copy

# getting the min area bounding for the rectangle
rect = cv.minAreaRect(largestContour)
points = cv.boxPoints(rect)  # getting the 4 points of the erea
points = np.int32(points)  # converting the points to int


h = math.sqrt(pow(points[0][0] - points[1][0], 2) +
              pow(points[0][1] - points[1][1], 2))  # getting the hight for the new image
w = math.sqrt(pow(points[1][0] - points[2][0], 2) +
              pow(points[1][1] - points[2][1], 2))  # getting the width for the new image


if (w > h):
    temp = h
    h = w
    w = temp
    output_pts = np.float32([[0, 0],
                             [w, 0],
                             [w, h],
                             [0, h]])  # in case the first point in the top left
else:
    output_pts = np.float32([[0, 0],
                             [0, h],
                             [w, h],
                             [w, 0]])  # in case the first point in the top right


transform_mat = cv.getPerspectiveTransform(
    np.float32(points), np.float32(output_pts))  # compute the prespective transform

res = cv.warpPerspective(
    img, transform_mat, (int(w), int(h)), flags=cv.INTER_LINEAR)  # apply the prespective transform to get the final image


plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))
plt.show()
cv.imwrite(output_dir, res)  # saving the new image
