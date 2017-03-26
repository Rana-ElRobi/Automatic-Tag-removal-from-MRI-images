# Load image ,get histogram , get fourier
import cv2
import dicom
import pylab
import numpy as np
import pylab as pl
import scipy.misc 
import numpy as np
import pymeanshift as pms
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
	original_image = cv2.imread('fourierfram1.png')
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

# Band stop Filter on clusterd image to remove tags
def BandStopFilter(imgSeg):
	pass
	return imgFiltered

# Inverse fourier transform
def invFourier():
	pass
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
	# Band Stop Filter
	dicFiltered = BandStopFilter(dicSegmented)
	# Invers Fourier transform 
	dicUnTagged = invFourier(dicFiltered)
	# Segment Left Venbterical 
	dicLeft = segmentMyocardial(dicUnTagged)

main ("/home/rana/Desktop/heart/Tagging&CineSets/No_NameTagging/IM-0001-0014.dcm")