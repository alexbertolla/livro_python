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
from numpy import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray


im = imread('../images/butterfly2.jpg')
im_gray = rgb2gray(im)
log_blobs = blob_log(im_gray, max_sigma=30, num_sigma=10, threshold=.1)
log_blobs[:, 2] = sqrt(2) * log_blobs[:, 2] #COMPUTE RADIUS IN THE 3RDCOLUMN

dog_blobs = blob_dog(im_gray, max_sigma=30, threshold=0.1)
dog_blobs[:, 2] = sqrt(2) * dog_blobs[:, 2]

doh_blob = blob_doh(im_gray, max_sigma=30, threshold=0.005)

list_blobs = [log_blobs, dog_blobs, doh_blob]
color, titles = ['yellow', 'lime', 'red'], ['Laplacian', 'Difference of Gaussian', 'Determinant of Hessian']
sequence = zip(list_blobs, color, titles)

fig, axes = pylab.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
axes = axes.ravel()

axes[0].imshow(im, interpolation='nearest')
axes[0].set_title('Original Image', size=15), axes[0].set_axis_off()

for idx, (blobs, color, title) in enumerate(sequence):
    axes[idx+1].imshow(im, interpolation='nearest')
    axes[idx + 1].set_title('Blobs with ' + title, size=15)
    for blob in blobs:
        y, x, row = blob
        col = pylab.Circle((x, y), row, color=color, linewidth=2, fill=False)
        axes[idx + 1].add_patch(col), axes[idx+1].set_axis_off()

pylab.tight_layout()
pylab.show()


print('FIM')