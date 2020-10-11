from google.colab import drive
drive.mount('/content/gdrive/')

#Importing the libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from google.colab.patches import cv2_imshow
from sklearn.mixture import GaussianMixture
import copy
import math

#Reading the image
img=cv2.imread('/content/drive/My Drive/dataset/example 1/20190813_001111.jpg')
#cv2_imshow(img)
example=1

#Converting to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2_imshow(gray)

# height, width, number of channels in image
height = gray.shape[0]
width = gray.shape[1] 

copy_gray=copy.deepcopy(gray)

if(example==1):
  im1=copy_gray[850:1800,500:width]
  im2=copy_gray[1800:height,500:width]
  #cv2_imshow(im1)
  #cv2_imshow(im2)

if (example==2):
  im1=copy_gray[690:1600,500:width]
  im2=copy_gray[1600:height,500:width]
  cv2_imshow(im1)

if(example==3):
  gray=gray[850:height,500:width]
  cv2_imshow(gray)  
  


#cv2_imshow(im2)

# height, width, number of channels in image
h1 = im1.shape[0]
w1 = im1.shape[1] 
h2 = im2.shape[0]
w2 = im2.shape[1] 


#Applying clustering algorithm to remove the mesh behind

#reshaping the image which is in gray scale
vectorized = im1.reshape((-1,1))
vectorized = np.float32(vectorized)

#Setting the criteria
criteria = (cv2.TERM_CRITERIA_EPS, 50, 0.001)

#number of clusters
K = 2
attempts=20
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
#center contains the mean intensity of the clusters
center = np.uint8(center)

res = center[label.flatten()]
result_image1 = res.reshape((im1.shape))

#Showing the resulting image
#cv2_imshow(result_image1)

ret, thresh2 = cv2.threshold(im2,128,255,cv2.THRESH_BINARY)
ret, thresh1 = cv2.threshold(result_image1,128,255,cv2.THRESH_BINARY)

final_image = np.concatenate((thresh1,thresh2), axis=0)
cv2_imshow(final_image)

edges = cv2.Canny(final_image, 75, 90, 11)
line = []
cv2_imshow(edges)

#Auto edge detection
v = np.median(final_image)
# apply automatic Canny edge detection using the computed median
lower = int(max(0, (1.0 - 0.33) * v))
upper = int(min(255, (1.0 + 0.33) * v))
blurred = cv2.GaussianBlur(final_image, (17, 17), 0)
edges = cv2.Canny(blurred, lower, upper)
cv2_imshow(edges)

if (example==1):
  e=copy.deepcopy(edges)
  kernel = np.ones((11,11), np.uint8)
  e= cv2.dilate(e, kernel, iterations=1) 
  kernel = np.ones((5,5), np.uint8)
  e= cv2.erode(e, kernel, iterations=1)
  cv2_imshow(e)
if(example==2):
  e=copy.deepcopy(edges)
  kernel = np.ones((7,7), np.uint8)
  e= cv2.dilate(e, kernel, iterations=1) 
  kernel = np.ones((3,3), np.uint8)
  e= cv2.erode(e, kernel, iterations=1)
  cv2_imshow(e)

final_image=cv2.cvtColor(final_image, cv2.COLOR_GRAY2RGB)

def distance (x1,x2,y1,y2):
  dist=np.sqrt(np.square(x1-x2)+np.square(y1-y2))
  return dist

if(example==1):
  #Applying HoughLinesP transformation
  lines = cv2.HoughLinesP(e,1,np.pi / 180,70,100,80,0)
  lines1= cv2.HoughLinesP(e,1,np.pi/180,50,100,20,5)


if(example==2):
  #Applying HoughLinesP transformation
  lines = cv2.HoughLinesP(e,1,np.pi / 180,70,100,75,5)
  lines1= cv2.HoughLinesP(e,1,np.pi/180,50,100,10,0)

for line1 in lines1:
  x1,y1,x2,y2=line1[0]
  if(distance(x1,x2,y1,y2)>75):
    continue
  cv2.line(final_image,(x1,y1),(x2,y2),(0,255,0),2,cv2.LINE_AA)

for line in lines:
  x1,y1,x2,y2=line[0]
  cv2.line(final_image,(x1,y1),(x2,y2),(0,0,255),6,cv2.LINE_AA)

cv2_imshow(final_image)
