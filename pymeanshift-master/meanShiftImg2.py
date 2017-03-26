# Hlper Link:
# https://github.com/fjean/pymeanshift
# Documentation
# https://github.com/fjean/pymeanshift/wiki
# To set up:
# http://linux-noosphere.blogspot.com.eg/2014/07/importerror-no-module-named-pymeanshift.html

# Workingggggggg 


import cv2
import numpy as np
import pymeanshift as pms
from matplotlib import pyplot as plt
#import scipy.io
#from PIL import Image
#from scipy.misc import toimage
im1 = '/home/rana/Desktop/liver/fourierfram1VERT.png'
im2 = "/home/rana/Desktop/heart/fourierfram1.png"
original_image = cv2.imread(im2)

(segmented_image, labels_image, number_regions) = pms.segment(original_image,spatial_radius=6,range_radius=4.5,min_density=50)
print ("number_regions")
print (number_regions)
print("labels_image")
print(labels_image.shape)
print("Original image dimenssions")
print(original_image.shape)
img_back= [original_image  , segmented_image,labels_image]
img_name = ['Original', 'Segmented','labels_image']
for i in xrange(3):
    plt.subplot(1,3,i+1),plt.imshow(img_back[i],cmap = 'gray')
    plt.title(img_name[i]), plt.xticks([]), plt.yticks([])
plt.show()
