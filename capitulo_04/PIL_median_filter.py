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
i = 1

pylab.figure(figsize=(10, 5))
pylab.subplots_adjust(hspace=0.8)

for prop_noise in np.linspace(0.05, 0.3, 3):
    n = int(im.width * im.height * prop_noise)
    x = np.random.randint(0, im.width, n)
    y = np.random.randint(0, im.height, n)
    for (x, y) in zip(x, y):
        #generate salt-and-pepper noise
        im.putpixel((x, y), ((0, 0, 0) if np.random.rand() < 0.5 else (255, 255, 255)))

    pylab.subplot(3, 4, i)
    pylab.title('Original image with ' + str(int(100 * prop_noise)) + '% noise added', size=10)
    pylab.imshow(im)
    i += 1

    for sz in [3, 7, 11]:
        im1 = im.filter(ImageFilter.MedianFilter(size=sz))
        pylab.subplot(3, 4, i)
        pylab.title('Output (Media Filter size=' + str(sz) + ')', size=10)
        pylab.imshow(im1)
        i += 1


pylab.show()

print('FIM')