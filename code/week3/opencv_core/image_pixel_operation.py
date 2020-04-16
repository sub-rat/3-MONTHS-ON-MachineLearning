import cv2
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread("/Users/berrybytes/Desktop/self/3-MONTHS-ON-MachineLearning/images/messi5.jpg")

# px = img[100,100]
# print(px)

# blue = img[100,100,0]
# print(blue)

# img[100,100] = [255,255,255]
# print(img[100,100])

print(img.shape)
print(img.size)

ball = img[1950:2260,2180:2510]
img[1950:2260,2180-800:2510-800] = ball
img[1950+50:2260+50,2180-1300:2510-1300] = ball
img[1950+100:2260+100,2180-1800:2510-1800] = ball

# b = img[:,:,0]
# img[:,:,2] = 0
# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

BLUE = [255,0,0]

img1 = img

replicate = cv2.copyMakeBorder(img1,200,200,200,200,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,200,200,200,200,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,200,200,200,200,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,200,200,200,200,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,200,200,200,200,cv2.BORDER_CONSTANT,value=BLUE)

plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

plt.show()