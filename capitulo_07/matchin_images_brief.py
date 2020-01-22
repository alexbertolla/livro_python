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
from skimage.feature import (match_descriptors, corner_peaks, corner_harris, plot_matches, BRIEF)
from skimage import transform
img1 = color.rgb2gray(imread('../images/lena.jpg'))
affine_trans = transform.AffineTransform(scale=(1.2, 1.2), translation=(0, -100))

print('FIM')