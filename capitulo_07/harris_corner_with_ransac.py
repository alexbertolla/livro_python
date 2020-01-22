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


def gaussian_weights(window_ext, sigma=1):
    y, x = np.mgrid[-window_ext:window_ext+1, -window_ext:window_ext+1]
    g_w = np.zeros(y.shape, dtype=np.double)
    g_w[:] = np.exp(-0.5, * (x**2 / sigma**2 + y**2 / sigma**2))
    g_w /= 2 * np.pi * sigma * sigma
    return g_w

def match_corner(coornidates, window_ext=3):
    row, col = np.round(coornidates).astype(np.intp)
    print(row+window_ext+1)
    window_original = image_original#[row - window_ext:row + window_ext + 1, col - window_ext:col + window_ext + 1, :]
#WEIGHT PIXELS DEPENDING ON THE DISTANCE TO THE CENTER PIXEL
#    weights = gaussian_weights(window_ext, 3)
#    weights = np.dstack((weights, weights, weights))
#COMPUTE THE SIM OF SQUARED DIFFERENCE TO ALL CORNERS IN THE WARPED IMAGE
#    SSDs = []
#    for coord_row, coord_col in coornidates_warped:
#        window_warped = image_warped[coord_row - window_ext:coord_row + window_ext + 1,
#                        coord_col - window_ext:coord_col + window_ext + 1, :]
#    if window_original.shape == window_warped.shape:
#        SSD = np.sum(weights * (window_original - window_warped)**2)
#        SSDs.append(SSD)

#    min_idx = np.argmin(SSDs) if len(SSDs) > 0 else -1
#    return coornidates_warped_subpix[min_idx] if min_idx >= 0 else [None]




#THE NEXT BLOCK SHOW HOW TO IMPLEMENT THE IMAGE MATCHING USING HARRIS CORNER FEATURES
temple = rgb2gray(img_as_float(imread('../images/lagarta/imagem_67_seg.jpg')))
image_original = np.zeros(list(temple.shape) + [3])
image_original[..., 0] = temple
gradient_row, gradient_col = (np.mgrid[0:image_original.shape[0],
                              0:image_original.shape[1]] / float(image_original.shape[0]))
image_original[..., 1] = gradient_row
image_original[..., 2] = gradient_col
image_original = rescale_intensity(image_original)
image_original_gray = rgb2gray(image_original)
affine_trans = AffineTransform(scale=(0.8, 0.9), rotation=0.1, translation=(120, -20))
image_warped = warp(image_original, affine_trans.inverse, output_shape=image_original.shape)
image_warped_gray = rgb2gray(image_warped)

#EXTRACT CORNERS USING THE HARRRIS CORNER MEASURE
coornidates = corner_harris(image_original_gray)
coornidates[coornidates > 0.01 * coornidates.max()] = 1
coornidates_original = corner_peaks(coornidates, threshold_rel=0.0001, min_distance=5)

coornidates = corner_harris(image_warped_gray)
coornidates[coornidates > 0.01 * coornidates.max()] = 1
coornidates_warped = corner_peaks(coornidates, threshold_rel=0.0001, min_distance=5)

#DETERMINE THE SUB-PIXEL CORNER POSITION
coornidates_original_subpix = corner_subpix(image_original_gray, coornidates_original, window_size=9)
coornidates_warped_subpix = corner_subpix(image_warped_gray, coornidates_warped, window_size=9)

#FIND THE CORESPONDENCES USING THE SIMPLE WEIGHT SUM OF THE SQUARED DIFFERENCES
source, destination = [], []
for coornidates in coornidates_warped_subpix:
    coornidates1 = match_corner(coornidates)


#VISUALIZE THE CORRESPONDENCE
fig, axes = pylab.subplots(nrows=2, ncols=1, figsize=(20, 15))
pylab.gray()



print('FIM')