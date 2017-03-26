# Helper Link 
# http://scikit-image.org/docs/dev/auto_examples/edges/plot_active_contours.html
import cv2
import numpy as np
import matplotlib.pyplot as plt
import dicom
import pylab
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# Test scipy version, since active contour is only possible
# with recent scipy version
import scipy
split_version = scipy.__version__.split('.')
if not(split_version[-1].isdigit()): # Remove dev string if present
        split_version.pop()
scipy_version = list(map(int, split_version))
new_scipy = scipy_version[0] > 0 or \
            (scipy_version[0] == 0 and scipy_version[1] >= 14)

path = "IM-0001-0101.dcm"
fourirPath = "heartfourierfram1.png"
ds=dicom.read_file(path)

pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)
#print(ds.pixel_array.shape)
#print(len(ds))
#<matplotlib.image.AxesImage object at 0x0162A530>
pylab.show()


img = ds.pixel_array
#img = cv2.imread(fourirPath)
#pylab.imshow(img, cmap=pylab.cm.bone)
#pylab.show()
#print (type(img))
s = np.linspace(0, 2*np.pi, 400)
# Four fourie
#x = 128.8 + 17*np.cos(s)
#y = 129.3 + 20*np.sin(s)

# There are very good for real heary images
x = 146 + 100*np.cos(s)
y = 124 + 100*np.sin(s)
init = np.array([x, y]).T

if not new_scipy:
    print('You are using an old version of scipy. '
          'Active contours is implemented for scipy versions '
          '0.14.0 and above.')

if new_scipy:
    # Good for outer
    #snake = active_contour(gaussian(img, 0.5),
    #                       init, alpha=0.05, beta=9, gamma=0.001)
    # This fitts the outer wall of the left atriam
    ## better
    snake = active_contour(gaussian(img, 2),
                           init, alpha=0.015, beta=10, gamma=0.001)
    # Fitts iner wall of left ventricl
    # boundries of the blood pool
    # Not bad alpha = 0.05 , gaussian= 2 , gamma=0.0001 or gamma=0.00015
    # Not bad too
    #snake = active_contour(gaussian(img, 2),
    #                       init, alpha=0.05, beta=1, gamma=0.00009)
    # Perfect 2 so far
    #snake = active_contour(gaussian(img, 0.5),
    #                       init, alpha=0.05, beta=5, gamma=0.001)
    # Perfect 1 so far for small muscel
    #snake = active_contour(gaussian(img, 0.5),
    #                       init, alpha=0.05, beta=1, gamma=0.001)
    # To find image pick
    #snake = active_contour(gaussian(img, 0.5),
    #                       init, alpha=0.01, beta=1, gamma=0.000001)
    
    #print (snake.shape)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plt.gray()
    ax.imshow(img)
    ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    plt.show()