from PIL import Image, ImageDraw
from numpy import array
from numpy import zeros
from numpy import ndarray
import numpy as np
import cv2
import random
from elementtree import ElementTree as ET
from matplotlib import pyplot as plt
import os as os

class Feature:
    def __init__(self, x, y):
        self.x = x
        self.y = y
class Predictor:
    def __init__(self, F_src, S, F_tgt, A):
        self.F_src = F_src
        self.S = S
        self.F_tgt = F_tgt
        self.A = A
    def _out_(self):
        print self.A, " "

windowSize = 50 # size of area where cannot be more than 1 predictor
def isPointInRect ((Px, Py), (Rx, Ry), windowSize):
    if (Px < Rx - windowSize or Px > Rx + windowSize):
        return False
    if (Py < Ry - windowSize or Py > Ry + windowSize):
        return False
    return True
def saccadeDistance (S1, S2):
    delta_x = S1.x-S2.x
    delta_y = S1.y-S2.y
    return np.sqrt(delta_x*delta_x+delta_y*delta_y)

m = 50  #number of images for teaching
N = 4 # number of attractors
SD = 50 #epsilon of saccade distance
folder_name = ['Giraffe/', 'Horse/', 'Dog/', 'Cat/']
predictors = []
Attractor = []
for i in range(N):
    Attractor.append([])

for attractor in range(N):
    path_to_image = "E:/Stick Animals/StickFigures/" + folder_name[attractor]
    for j in os.listdir(path_to_image):
        img = cv2.imread(path_to_image + j)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = np.float32(gray_img)
        blockSize = 2;
        kSize = 3;
        dst = cv2.cornerHarris(gray_img, blockSize, kSize, 0.04)
        features = []
        corners =  np.where (dst > 0.2*dst.max())
        for i in range(len(corners[0])):
            x = corners[0][i]
            y = corners[1][i]
            img[(x, y)] = [0, 255, 0]
            if len (features) == 0:
                features.append(Feature(y, x))
            else:
                f = False
                for j in features:
                    if isPointInRect((x, y), (j.y, j.x), windowSize) == True:
                        f = True
                        break
                if f == False:
                    features.append(Feature(y, x))
        for i in range (len(features)):
            img = cv2.circle (img, (features[i].x, features[i].y), 10, (0,255,0), 3)
            # print features[i].x, features[i].y
            # print '\n'
            l = i
            while (l == i):
                l = random.randint(0, len(features)-1)
                S = Feature (features[l].x-features[i].x, features[l].y-features[i].y)
                if l != i:
                    predictors.append(Predictor(features[i], S, features[l], attractor));
        # cv2.imwrite(path_to_image + "1_1.png", img)
        # cv2.imshow('', img)
        # if cv2.waitKey(0):
        #     cv2.destroyAllWindows()

root = ET.Element('root')
for i in range (len(predictors)):
    predictor_tag = ET.SubElement(root, 'Predictor', number = str(i))
    F_src_tag = ET.SubElement(predictor_tag, 'F_src')
    x_tag = ET.SubElement(F_src_tag, 'x').text = str(predictors[i].F_src.x)
    y_tag = ET.SubElement(F_src_tag, 'y').text = str(predictors[i].F_src.y)
    Saccade_tag = ET.SubElement(predictor_tag, 'S')
    x_tag = ET.SubElement(Saccade_tag, 'x').text = str(predictors[i].S.x)
    y_tag = ET.SubElement(Saccade_tag, 'y').text = str(predictors[i].S.y)
    F_tgt_tag = ET.SubElement(predictor_tag, 'F_src')
    x_tag = ET.SubElement(F_tgt_tag, 'x').text = str(predictors[i].F_tgt.x)
    y_tag = ET.SubElement(F_tgt_tag, 'y').text = str(predictors[i].F_tgt.y)
    A_tag = ET.SubElement(predictor_tag, 'Attractor').text = str(predictors[i].A)
tree = ET.ElementTree(root)
tree.write("E:/Stick Animals/predictors.xml")
# features = []
# path_to_image = "E:/Stick Animals/StickFigures/" + folder_name[0]
# img = cv2.imread(path_to_image+"15.png")
# # searching features in input image
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# grayimg = np.float32(gray_img)
# blockSize = 2;
# kSize = 3;
# dst = cv2.cornerHarris(gray_img, blockSize, kSize, 0.04)
# features = []
# corners =  np.where (dst > 0.2*dst.max())
# for i in range(len(corners[0])):
#     x = corners[0][i]
#     y = corners[1][i]
#     img[(x, y)] = [0, 255, 0]
#     if len (features) == 0:
#         features.append(Feature(y, x))
#     else:
#         f = False
#         for j in features:
#             if isPointInRect((x, y), (j.y, j.x), windowSize) == True:
#                 f = True
#                 break
#         if f == False:
#             features.append(Feature(y, x))
# for i in features:
#     img = cv2.circle (img, (i.x, i.y), 10, (0,255,0), 3)
#
# n = len(predictors)
# for i in range(N):              # making matrix of attrators
#     for j in range (n):
#         Attractor[i].append(-1)
#
# for i in range (N):             #filling the matrix of attrators
#     for j in range(n):
#         if predictors[j].A == i:
#             Attractor[i][j] = 1
# # matrix of Hopfield network
# J = np.zeros((n, n))
# for i in range (N):
#     A_mu = np.matrix(Attractor[i])
#     J = J + np.multiply(A_mu.T, A_mu)
# for i in range(n):
#     J[i,i] = 0
# J = J/N
#
# a = n/2
# A = []
# for i in range(n):
#     A.append(-1)
# while (a > 0):                  #generation of random vector of a positive elements
#     j = random.randint(0, n-1)
#     if (A[j] == -1):
#         A[j] = 1
#         a = a - 1
# RP = [0]
# SP = []
# t = n/2 # treshold
# fovea = Feature(0, 0)
# fovea = features[0]
#
# F_srcSize = 50 #size of area where will be found source features
# SSS = 40 #size of successful saccade area, i.e. succade which lead to some target points
# while 1 == 1:
#     RP = []
#     for i in range(n):
#         #img = cv2.putText(img, str(A[i]), (predictors[i].F_src.x, predictors[i].F_src.y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
#         if A[i] == 1:
#             if isPointInRect((predictors[i].F_src.x, predictors[i].F_src.y), (fovea.x, fovea.y), F_srcSize):
#                 sCurrent = predictors[i].S
#                 for j in features:
#                     if (isPointInRect((sCurrent.x + fovea.x, sCurrent.y+fovea.y), (j.x, j.y), SSS) == True):
#                         RP.append(i)
#     # print RP
#     for i in RP:
#         img = cv2.circle (img, (predictors[i].F_src.x, predictors[i].F_src.y), 10, (0,0,255), 3)
#     img = cv2.circle (img, (fovea.x, fovea.y), 10, (255,0,0), 3)
#     img = cv2.rectangle(img, (fovea.x-F_srcSize, fovea.y+F_srcSize), (fovea.x+F_srcSize, fovea.y-F_srcSize), (0,0,255), 1)
#     cv2.imshow('', img)
#     if cv2.waitKey(0):
#         cv2.destroyAllWindows()
#
#     AA = A
#     SP = []
#     windowSize = 40
#     while len (RP)!=0:
#         # starting saccade selection with checking which of 'em lead to target points in input img
#         sUsage = [0 for i in predictors]
#         for i in RP:
#             sCurrent = predictors[i].S
#             for j in RP:
#                 if (saccadeDistance(sCurrent, predictors[j].S) < SD):
#                     sUsage[i] = sUsage[i] + 1
#         S_best = np.argmax(sUsage)
#         fovea.x = fovea.x + predictors[S_best].S.x
#         fovea.y = fovea.y + predictors[S_best].S.y
#         # img = cv2.arrowedLine(img, (fovea.x-predictors[S_best].S.x, fovea.y-predictors[S_best].S.y), (fovea.x, fovea.y), (0,0,0))
#         # img = cv2.circle (img, (fovea.x, fovea.y), 10, (0,100,255), 3)
#         # img = cv2.rectangle(img, (fovea.x-windowSize, fovea.y+windowSize), (fovea.x+windowSize, fovea.y-windowSize), (0,0,255), 1)
#         # cv2.imshow('', img)
#         # if cv2.waitKey(0):
#         #     cv2.destroyAllWindows()
#
#         # selection of successful predictors
#
#         F_tgtSize = 50
#         usedPredictors = []
#         s = 5 # number of predictors which will be activated with SP
#         for j in RP:
#             if saccadeDistance(predictors[S_best].S, predictors[j].S) < SD:
#                 if (isPointInRect((predictors[j].F_tgt.x, predictors[j].F_tgt.y), (fovea.x, fovea.y), F_tgtSize)):
#                     # img = cv2.arrowedLine(img, (predictors[j].F_src.x, predictors[j].F_src.y), (predictors[j].F_tgt.x, predictors[j].F_tgt.y), (255,255,0))
#                     # img = cv2.circle (img, (predictors[j].F_tgt.x, predictors[j].F_tgt.y), 10, (128,0,128), 3)
#                     # cv2.imshow('', img)
#                     # if cv2.waitKey(0):
#                     #     cv2.destroyAllWindows()
#                     SP.append(j)
#                     indexes_of_predictors = []
#                     for i in range(n):
#                         if predictors[i].A == predictors[j].A: #indexes in predictors equals indexes in A, so we can find
#                             indexes_of_predictors.append(i)    # predictors of the same attractor and activate s random of them
#                     for i in range(s):
#                         k = random.randint(0, len(indexes_of_predictors)-1)
#                         A[indexes_of_predictors[k]] = 1
#                         del indexes_of_predictors[k]
#                 else:
#                     AA[j] = -1
#                 usedPredictors.append(j)
#         for i in usedPredictors:
#             RP.remove(i)
#     # update network
#     for i in range(n):
#         S = 0
#         for j in range (n):
#             if (i != j):
#                 S = S + J[i][j]*(AA[j]+1)
#                 S = S/2
#     S = 0
#     mn = np.power(2*n, 3)
#     i_min = 0
#     for i in range (N):
#         for j in range(len(Attractor[i])):
#             S = S + (Attractor[i][j]-A[i][j])*(Attractor[i][j]-A[i][j])
#         if (S < mn):
#             mn = S
#             i_min = i
#     if (S < t):
#         print i_min
#         break