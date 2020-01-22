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


image = imread('../images/chess.jpg') #RGB IMAGE

image_gray = rgb2gray(image)
coordinates = corner_harris(image_gray, k=0.001)
image[coordinates > 0.01 * coordinates.max()] = [255, 0, 255]
pylab.figure(figsize=(20, 20))
pylab.imshow(image), pylab.axis('off'), pylab.show()


print('FIM')