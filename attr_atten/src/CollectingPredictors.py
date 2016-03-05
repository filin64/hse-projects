from PIL import Image, ImageDraw
from numpy import array
from numpy import zeros
from numpy import ndarray
import numpy as np
import cv2
from matplotlib import pyplot as plt

class Feature:
    def __init__(self, x, y):
        self.x = x
        self.y = y
windowSize = 20
def isPointInRect (Px, Py, Rx, Ry):
    if (Px < Rx - windowSize or Px > Rx + windowSize):
        return False
    if (Py < Ry - windowSize or Py > Ry + windowSize):
        return False
    return True

path_to_image = "E:/Stick Animals/"

img = cv2.imread(path_to_image+"1.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img = np.float32(gray_img)
blockSize = 2;
kSize = 3;
dst = cv2.cornerHarris(gray_img, blockSize, kSize, 0.04)
predictors = []
corners =  np.where (dst > 0.2*dst.max())
for i in range(len(corners[0])):
    x = corners[0][i]
    y = corners[1][i]
    img[(x, y)] = [0, 255, 0]
    if len (predictors) == 0:
        predictors.append(Feature(x, y))
    else:
        f = False
        for j in predictors:
            if isPointInRect(x, y, j.x, j.y) == True:
                f = True
                break
        if f == False:
            predictors.append(Feature(x, y))

for i in range (len(predictors)):
    img = cv2.circle (img, (predictors[i].y, predictors[i].x), 10, (0,255,0), 3)
cv2.imwrite(path_to_image + "1_1.png", img)
cv2.imshow('', img)
if cv2.waitKey(0):
    cv2.destroyAllWindows()
