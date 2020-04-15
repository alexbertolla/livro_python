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

im = Image.open('../images/mandrill.jpg')

pylab.figure(figsize=(15, 6))
i = 1

for radius in range(1, 4):
    im1 = im.filter(ImageFilter.GaussianBlur(radius))
    pylab.subplot(1, 3, i)
    pylab.title('radius = ' + str(round(radius, 2)))
    pylab.imshow(im1)
    i += 1
pylab.suptitle('PIL Gaussian Blur with different Radius', size=20)
pylab.show()

print('FIM')