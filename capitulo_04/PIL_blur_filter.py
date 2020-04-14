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

#script página 160

im = Image.open('../images/mandrill.jpg')
pylab.figure(figsize=(10, 25))
pylab.subplots_adjust(hspace=0.5)
pylab.subplot(4, 2, 1)
pylab.title('Original image')
pylab.imshow(im)

pylab.subplot(4, 2, 2)
pylab.title('Original image blured')
pylab.imshow(im.filter(ImageFilter.BLUR))

i = 3
for prop_noise in np.linspace(0.05, 0.3, 3):
    #choose 5000 random locations inside the image
    n = int(im.width * im.height * prop_noise)
    x = np.random.randint(0, im.width, n)
    y = np.random.randint(0, im.height, n)
    for (x, y) in zip(x, y):
        #generate salt-and-pepper noise
        im.putpixel((x, y), ((0, 0, 0) if np.random.rand() < 0.5 else (255, 255, 255)))

    im1 = im.filter(ImageFilter.BLUR)

    pylab.subplot(4, 2, i)
    pylab.title('Original image with ' + str(100 * prop_noise) + '% added noise')
    pylab.imshow(im)


    pylab.subplot(4, 2, 1+i)
    pylab.title('Image with ' + str(100 * prop_noise) + '% added noise blured')
    pylab.imshow(im1)
    i += 2


pylab.show()

print('FIM')