# Load image ,get histogram , get fourier
import cv2
import dicom
import pylab
#import imutils
import argparse
import scipy.misc 
import numpy as np
import pylab as pl
import numpy as np
import pymeanshift as pms
from skimage import measure
import matplotlib.pyplot as plt

#from matplotlib import pyplot as plt

# Load Dicom image
# Helper Link
# http://pydicom.readthedocs.io/en/stable/viewing_images.html
def loadDICOM(imagePath):
	ds=dicom.read_file(imagePath)
	print(ds.pixel_array.shape)
	print(len(ds))
	#<matplotlib.image.AxesImage object at 0x0162A530>
	pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)
	pylab.show()
	return ds.pixel_array

# Helper Link
# http://docs.opencv.org/trunk/d1/db7/tutorial_py_histogram_begins.html
def checkHistogram(imgArray):
	hist = cv2.calcHist([imgArray],[0],None,[256],[0,256])
	plt.plot(hist,color = 'gray')
	plt.xlim([0,256])
	plt.ylim([0,800])
	plt.show()
	return hist

# Transform image to fourier transform
# Helper link
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
def fourier(imgPixels):
	f = np.fft.fft2(imgPixels)
	fshift = np.fft.fftshift(f)
	magnitude_spectrum = 20*np.log(np.abs(fshift))
	# Save image
	scipy.misc.imsave('fourierfram1.png', magnitude_spectrum)	
	# Plot frequency donaim VS Spatial Domain
	plt.subplot(121),plt.imshow(imgPixels, cmap = 'gray')
	plt.title('Input Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
	plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
	plt.show()
	return magnitude_spectrum

# Cluster fourier image 
# Hlper Link:
# https://github.com/fjean/pymeanshift
# Documentation
# https://github.com/fjean/pymeanshift/wiki
# To set up:
# http://linux-noosphere.blogspot.com.eg/2014/07/importerror-no-module-named-pymeanshift.html
def meanShift():
	original_image = cv2.imread('heartfourierfram1.png')
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
	return segmented_image

# Helper Link
# https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.sparse.csgraph.connected_components.html
def ConnComp(imgTarget):
	comp , lables = scipy.sparse.csgraph.connected_components(imgTarget,directed=True,connection='strong',return_labels=True)
	print (comp)
	print (lables)
	#

	L = measure.label(imgTarget)
	#

	print "Number of components:", np.max(L)
	return comp,lables

# Helper Link
# http://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
def findContours(targetimg):
	# load the query image, compute the ratio of the old height
	# to the new height, clone it, and resize it
	image = cv2.imread(targetimg)
	ratio = image.shape[0] / 300.0
	orig = image.copy()
	image = imutils.resize(image, height = 300)
	# convert the image to grayscale, blur it, and find edges
	# in the image
	gray = cv2.cvtColor(targetimg, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 11, 17, 17)
	edged = cv2.Canny(gray, 30, 200)
	# find contours in the edged image, keep only the largest
	# ones, and initialize our screen contour
	(cnts, _) = cv2.findContours(targetimg.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
	screenCnt = None
	return contours


# Band stop Filter on clusterd image to remove tags
def BandStopFilter(imgSeg):
	pass
	return imgFiltered

# Inverse fourier transform
def invFourier(imgF):
	# Return complex 2d array image
	spatialImg = numpy.fft.fft2(imgF)
	return spatialImg

def main(fileName):
	# Read Dicom as pixels array
	dicFram = loadDICOM(fileName)
	# Check Histogram
	dicHist = checkHistogram(dicFram)
	# Transform to Frequency Domain
	dicFourier = fourier(dicFram)
	# cluster peaks in frequancy domain by meanshift
	dicSegmented = meanShift()
	# Get connected components
	dicComponents = ConnComp(dicFourier)
	# Cluster by finding contours around edges of fourier
	#dicCountours = findContours("heartfourierfram1.png")
	# Band Stop Filter
	#dicFiltered = BandStopFilter(dicSegmented)
	# Invers Fourier transform 
	#dicUnTagged = invFourier(dicFiltered)
	# Segment Left Venbterical 
	#dicLeft = segmentMyocardial(dicUnTagged)

main ("/home/rana/Desktop/heart/Tagging&CineSets/No_NameTagging/IM-0001-0014.dcm")
