import numpy as np
from skimage import data, img_as_float, img_as_ubyte, exposure, io, color
from skimage.io import imread
from skimage.exposure import cumulative_distribution
from skimage.restoration import denoise_bilateral, denoise_nl_means, estimate_sigma
from skimage.measure import compare_psnr
from skimage.util import random_noise
from skimage.color import rgb2gray
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage, misc
import matplotlib.pylab as pylab


#script página 151

img = imread('../images/tajmahal.jpg')
img = rgb2gray(img)

img_eq = exposure.equalize_hist(img)
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

images = [img, img_eq, img_adapteq]
titles = ['Original input', 'After histogram equalization', 'After adaptative histogram equalization']

pylab.figure(figsize=(10, 5))
pylab.gray()
pylab.subplots_adjust(hspace=0.7)

pylab.subplot(3, 2, 1), pylab.title(titles[0]), pylab.imshow(images[0])
pylab.subplot(3, 2, 2), pylab.title(titles[0]), pylab.hist(images[0].ravel())

pylab.subplot(3, 2, 3), pylab.title(titles[1]), pylab.imshow(images[1])
pylab.subplot(3, 2, 4), pylab.title(titles[1]), pylab.hist(images[1].ravel())

pylab.subplot(3, 2, 5), pylab.title(titles[2]), pylab.imshow(images[2])
pylab.subplot(3, 2, 6), pylab.title(titles[2]), pylab.hist(images[2].ravel())

pylab.show()



pylab.show()

print('FIM')