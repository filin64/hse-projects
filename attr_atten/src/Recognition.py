from PIL import Image, ImageDraw
from numpy import array
from numpy import zeros
from numpy import ndarray
import numpy as np
import cv2
import random
from elementtree import ElementTree as ET

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

# function defines if some point is laying within some square area
def isPointInRect ((Px, Py), (Rx, Ry), predictorArea):
    if (Px < Rx - predictorArea or Px > Rx + predictorArea):
        return False
    if (Py < Ry - predictorArea or Py > Ry + predictorArea):
        return False
    return True
# Euclidean 2-D distance
def distance (S1, S2):
    delta_x = S1.x-S2.x
    delta_y = S1.y-S2.y
    return np.sqrt(delta_x*delta_x+delta_y*delta_y)
def sign (a):
    if (a < 0):
        return -1
    else:
        return 1

predictorArea = 50 # area size containing only 1 predictor
m = 50  #number of images for teaching
N = 4 # number of attractors
SD = 50 #epsilon of saccade distance
result = -1 # the number of attractor
folder_name = ['Giraffe/', 'Horse/', 'Dog/', 'Cat/']
predictors = []
# reading predictors
tree = ET.parse('E:/Stick Animals/predictors.xml')
root = tree.getroot()
for predictor in root:
    F_src = Feature(int(predictor[0][0].text), int(predictor[0][1].text))
    S = Feature (int(predictor[1][0].text), int(predictor[1][1].text))
    F_tgt = Feature(int(predictor[2][0].text), int(predictor[2][1].text))
    A = int(predictor[3].text)
    predictors.append(Predictor(F_src, S, F_tgt, A))
Attractor = [] # N x n matrix containing attractors
for i in range(N):
    Attractor.append([])
features = [] # a list containing info about found picture's features
path_to_image = "E:/Stick Animals/StickFigures/" + folder_name[1]
img = cv2.imread(path_to_image+"1.png")
# searching features in input image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayimg = np.float32(gray_img)
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
            if isPointInRect((x, y), (j.y, j.x), predictorArea) == True:
                f = True
                break
        if f == False:
            features.append(Feature(y, x))

# marking features on picture
for i in features:
    img = cv2.circle (img, (i.x, i.y), 10, (0,255,0), 3)

n = len(predictors)
for i in range(N):              # making matrix of attrators
    for j in range (n):
        Attractor[i].append(-1)

for i in range (N):             #filling the matrix of attrators
    for j in range(n):
        if predictors[j].A == i:
            Attractor[i][j] = 1
# matrix of Hopfield network
J = np.zeros((n, n))
for i in range (N):
    A_mu = np.matrix(Attractor[i])
    J = J + A_mu.T*A_mu
for i in range(n):
    J[i,i] = 0
J = J/N

a = 100 #number of random chosen predictors which are 1
A = []  #potential Attractor
for i in range(n):
    A.append(-1)
while (a > 0):                  #generation of random vector of a positive elements
    j = random.randint(0, n-1)
    if (A[j] == -1):
        A[j] = 1
        a = a - 1
for i in range(n):
    if A[i] == 1:
        img = cv2.circle (img, (predictors[i].F_src.x, predictors[i].F_src.y), 10, (133,133,133), 3)
cv2.imshow('All active predictors', img)
if cv2.waitKey(0):
    cv2.destroyAllWindows()
RP = []
SP = []
NR = [] # Necessary resources
R = np.zeros((n, 1))
for i in range(n):
    NR.append(random.randint(10,20))
NR = np.matrix(NR).T
t = 0.3 # treshold of attractor basin
fovea = Feature(0, 0)
F_srcSize = 100 #size of area where will be found source features
F_tgtSize = 50 #size of area where shoul be target point
for step in range (10):
    RP = []
    # setting fovea at random position
    j = random.randint(0, len(features)-1)
    fovea = features[j]
    for i in range(n):
        #img = cv2.putText(img, str(A[i]), (predictors[i].F_src.x, predictors[i].F_src.y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
        if A[i] == 1:
            # if F_src belongs to fovea view
            if isPointInRect((predictors[i].F_src.x, predictors[i].F_src.y), (fovea.x, fovea.y), F_srcSize):
                sCurrent = predictors[i].S
                for j in features:
                    # ...and if saccade of this predictor lead to some feature
                    if (isPointInRect((sCurrent.x + fovea.x, sCurrent.y+fovea.y), (j.x, j.y), F_tgtSize) == True):
                        RP.append(i)
    for i in RP:
        img = cv2.circle (img, (predictors[i].F_src.x, predictors[i].F_src.y), 10, (0,0,255), 3)
    img = cv2.circle (img, (fovea.x, fovea.y), 10, (255,0,0), 3)
    img = cv2.rectangle(img, (fovea.x-F_srcSize, fovea.y+F_srcSize), (fovea.x+F_srcSize, fovea.y-F_srcSize), (0,0,255), 1)
    cv2.imshow('Relevant Predictors and fovea', img)
    if cv2.waitKey(0):
        cv2.destroyAllWindows()

    AA = A
    SP = []
    predictorArea = 40
    while len (RP)!=0:
        sUsage = [0 for i in predictors] # list of saccade voting
        for i in RP:
            sCurrent = predictors[i].S
            for j in RP:
                # if distance between Saccade less then some constant means that they lead to the same point
                if (distance(sCurrent, predictors[j].S) < SD):
                    sUsage[i] = sUsage[i] + 1
        S_best = np.argmax(sUsage)
        fovea.x = fovea.x + predictors[S_best].S.x
        fovea.y = fovea.y + predictors[S_best].S.y
        img = cv2.arrowedLine(img, (fovea.x-predictors[S_best].S.x, fovea.y-predictors[S_best].S.y), (fovea.x, fovea.y), (0,0,0))
        img = cv2.circle (img, (fovea.x, fovea.y), 10, (0,100,255), 3)
        cv2.imshow('Moving fovea through the best saccade and fovea' + str(predictors[S_best].S.x) + ', ' + str(predictors[S_best].S.y), img)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()

        # selection of successful predictors


        usedPredictors = [] # list of checked predictors
        s = 5 # number of predictors which will be activated with SP
        for j in RP:
            # is current saccade is S_best
            if distance(predictors[S_best].S, predictors[j].S) < SD:
                # if target area is within fovea view
                if (isPointInRect((predictors[j].F_tgt.x, predictors[j].F_tgt.y), (fovea.x, fovea.y), F_tgtSize)):
                    img = cv2.arrowedLine(img, (predictors[j].F_src.x, predictors[j].F_src.y), (predictors[j].F_tgt.x, predictors[j].F_tgt.y), (255,255,0))
                    img = cv2.circle (img, (predictors[j].F_tgt.x, predictors[j].F_tgt.y), 10, (128,0,128), 3)
                    cv2.imshow('Found target points', img)
                    if cv2.waitKey(0):
                        cv2.destroyAllWindows()
                    SP.append(j)
                    indexes_of_predictors = []
                    for i in range(n):
                        if predictors[i].A == predictors[j].A: #indexes in predictors equals indexes in A, so we can find
                            indexes_of_predictors.append(i)    # predictors of the same attractor and activate s random of them
                    for i in range(s):
                        k = random.randint(0, len(indexes_of_predictors)-1)
                        A[indexes_of_predictors[k]] = 1
                        del indexes_of_predictors[k]
                else:
                    AA[j] = -1
                usedPredictors.append(j)
        for i in usedPredictors:
            RP.remove(i)
    # update network
    if (step != 0):
        for i in range(n):
            A[i] = sign(R.item(i, 0)-NR.item(i, 0))
            if A[i] == 1:
                img = cv2.circle(img, (predictors[i].F_src.x, predictors[i].F_src.y), 10, (50, 133, 240), 2)
    cv2.imshow('All active predictors after step ' + str(step), img)
    if cv2.waitKey(0):
        cv2.destroyAllWindows()
    BB = np.matrix(AA).T+1
    R = J*BB
    R = R * 1/2
    for i in range(N):
        # calc the card of active predictors belonging to the attractor i
        activePredictorsNum = 0
        for j in predictors:
            if j.A == i:
                activePredictorsNum = activePredictorsNum + 1
        # intersection of set A with set of the attractor to find common active predictors
        commonActivePredictorsNum = 0
        for j in Attractor[i]:
           if j == A[j] and j == 1:
               commonActivePredictorsNum = commonActivePredictorsNum + 1
        if commonActivePredictorsNum / activePredictorsNum > t:
            result = i
            break
    if result != -1:
        break
print result