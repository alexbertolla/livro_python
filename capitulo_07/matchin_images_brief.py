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


img2 = transform.warp(img1, affine_trans)
img3 = transform.rotate(img1, 25)

coords1, coords2, coords3 = corner_harris(img1), corner_harris(img2), corner_harris(img3)

coords1[coords1 > 0.01 * coords1.max()] = 1
coords2[coords2 > 0.01 * coords2.max()] = 1
coords3[coords3 > 0.01 * coords3.max()] = 1

keypoints1 = corner_peaks(coords1, min_distance=5)
keypoints2 = corner_peaks(coords2, min_distance=5)
keypoints3 = corner_peaks(coords3, min_distance=5)


extractor = BRIEF()
extractor.extract(img1, keypoints1)
keypoints1, descriptors1 = keypoints1[extractor.mask], extractor.descriptors

extractor.extract(img2, keypoints2)
keypoints2, descriptors2 = keypoints2[extractor.mask], extractor.descriptors

extractor.extract(img3, keypoints3)
keypoints3, descriptors3 = keypoints3[extractor.mask], extractor.descriptors

pylab.subplot(211), pylab.imshow(descriptors1)
pylab.subplot(212), pylab.imshow(descriptors2)
pylab.show()
print(descriptors1.shape, descriptors2.shape)

matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
matches13 = match_descriptors(descriptors1, descriptors3, cross_check=True)

fig, axes = pylab.subplots(nrows=2, ncols=1)
pylab.gray()

plot_matches(axes[0], img1, img2, keypoints1, keypoints2, matches12)
axes[0].axis('off'), axes[0].set_title('Original Image vs. Transformed Image 2')

plot_matches(axes[1], img1, img3, keypoints1, keypoints3, matches13)
axes[1].axis('off'), axes[1].set_title('Original Image vs. Transformed Image 3')

pylab.show()


print('FIM')