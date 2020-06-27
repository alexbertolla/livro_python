from skimage import color, data, restoration, filters
from skimage.io import imread
from scipy.signal import convolve2d as conv2
import numpy as np
from matplotlib import pylab

im_o = imread('../images/69015.jpg')
#im_o = color.rgb2gray(im_o)

im_g = filters.gaussian(im_o, sigma=5)

n = 4
psf = np.ones((n, n)) / n**2
im_w, _ = restoration.unsupervised_wiener(color.rgb2gray(im_g), psf)

fig, axes = pylab.subplots(nrows=1, ncols=3, figsize=(5, 5), sharex=True, sharey=True)
pylab.gray()

axes[0].imshow(im_o), axes[0].axis('off'), axes[0].set_title('Origianl Images', size=20)
axes[1].imshow(im_g), axes[1].axis('off'), axes[1].set_title('Gaussian Blurred Image', size=20)
axes[2].imshow(im_w), axes[2].axis('off'), axes[2].set_title('Wiener Filtered Image', size=20)

pylab.show()

print('FIM')