import cv2
print(cv2.__version__)
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/Users/berrybytes/Desktop/self/3-MONTHS-ON-MachineLearning/images/messi5.jpg",0)
# cv2.imshow('image',img)
# cv2.waitKey('a')
# cv2.destroyAllWindows()

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()