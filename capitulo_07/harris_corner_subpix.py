#DEFEAULT IMPORTS
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from PIL.ImageChops import add, subtract, multiply, difference, screen
import PIL.ImageStat as stat
from skimage.io import imread, imsave, imshow, show, imread_collection, imshow_collection
from skimage import color, exposure, img_as_float, data, viewer
from skimage.transform import SimilarityTransform, warp, swirl
from skimage.util import invert, random_noise, montage
import matplotlib.image as mpimg
from matplotlib import pylab as pylab
from scipy.ndimage import affine_transform, zoom
from scipy import misc

#CHAPTER IMPORTS
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import AffineTransform, resize
import cv2
from skimage.exposure import rescale_intensity
from skimage.measure import ransac
from skimage.color import rgb2gray


image = imread('../images/pyramids2.jpg') #RGB IMAGE
image_gray = rgb2gray(image)
coordinates = corner_harris(image_gray, k=0.001)

coordinates[coordinates > 0.03 * coordinates.max()] = 255 # THRESHOLD FOR AN OPTIMAL VALUE, DEPENDS ON THE IMAGE
corner_coordinates = corner_peaks(coordinates)
coordinates_subpix = corner_subpix(image_gray, corner_coordinates, window_size=11)
pylab.figure(figsize=(20, 20))

pylab.subplot(211), pylab.imshow(coordinates, cmap='inferno')
pylab.plot(coordinates_subpix[:, 1], coordinates_subpix[:, 0], 'r', markersize=5, label='Subpixel')
pylab.legend(prop={'size': 5})
pylab.axis('off')

pylab.subplot(212), pylab.imshow(image, interpolation='nearest')
pylab.plot(corner_coordinates[:, 1], corner_coordinates[:, 0], 'bo', markersize=5)
pylab.plot(coordinates_subpix[:, 1], coordinates_subpix[:, 0], 'r+', markersize=10)
pylab.axis('off')
pylab.tight_layout()
pylab.show()

print('FIM')