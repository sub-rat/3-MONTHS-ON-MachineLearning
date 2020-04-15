import cv2
import numpy as np 
from matplotlib import pylab as plt

# CREATE A BALCK IMAGE
img = np.zeros((512,512,3),np.uint8)

# DRAWING LINE
cv2.line(img= img,pt1= (0,0), pt2= (511,511),color=(255,0,0),thickness=5)

# DRAWING RECTANGLE
cv2.rectangle(img=img,pt1= (0,0), pt2= (511,511),color=(255,0,0),thickness=3)

# DRAWING CIRCLE
cv2.circle(img=img,center=(447,63),radius=63,color=(255,0,0),thickness=-1)

# DRAWING ELLIPSE
cv2.ellipse(img=img,center=(256,256), axes=(100,50),angle= 0,startAngle= 0,endAngle=180,color=(255,0,0),thickness=-1)

# DRAWING POLLYGON
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
img = cv2.polylines(img,[pts],True,(0,255,255))


font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)

#SHOW IMAGE
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()