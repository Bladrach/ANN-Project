import cv2
import numpy as np
from matplotlib import pyplot as plt

dim = (100, 100)
image1 = cv2.imread('test\\7_Tomato___Target_Spot\\0b126ce6-af82-477f-8f4e-1de79d84a6dd___Com.G_TgS_FL 8294.JPG', -1)
image1 = cv2.resize(image1, dim, interpolation = cv2.INTER_AREA)
gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
image2 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)
huMoments = cv2.HuMoments(cv2.moments(gray))
print(huMoments)
cv2.imshow('asd', image1)



lu1=image2[:,:,0].flatten()
plt.subplot(1,3,1)
plt.hist((lu1/lu1.max())*360 ,bins=8,range=(0.0,360.0),histtype='stepfilled', color='r', label='Hue')
plt.title("Hue")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()

lu2=image2[:,:,1].flatten()
plt.subplot(1,3,2)
plt.hist(lu2/255,bins=8,range=(0.0,1.0),histtype='stepfilled', color='g', label='Saturation')
plt.title("Saturation")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()

lu3=image2[:,:,2].flatten()
plt.subplot(1,3,3)
plt.hist(lu3*255,bins=8,range=(0.0,255.0),histtype='stepfilled', color='b', label='Value')
plt.title("Value")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()

plt.show()