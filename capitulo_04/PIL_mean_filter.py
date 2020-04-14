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
#choose 5000 random locations inside the image
prop_noise = 0.05
n = int(im.width * im.height * prop_noise)
x = np.random.randint(0, im.width, n)
y = np.random.randint(0, im.height, n)
for (x, y) in zip(x, y):
    #generate salt-and-pepper noise
    im.putpixel((x, y), ((0, 0, 0) if np.random.rand() < 0.5 else (255, 255, 255)))

pylab.figure(figsize=(10, 25))
pylab.subplots_adjust(hspace=0.5)
pylab.subplot(1, 3, 1)
pylab.title('Original image')
pylab.imshow(im)

for n in [3, 5]:
    box_blur_kernel = np.reshape(np.ones(n*n),(n,n))/(n*n)
    im1 = im.filter(ImageFilter.Kernel((n,n), box_blur_kernel.flatten()))
    pylab.subplot(1, 3, (2 if n==3 else 3))
    pylab.title('Blured image with kernel size = ' + str(n) + 'x' + str(n))
    pylab.imshow(im1)

pylab.suptitle('PIL Mean Filter (Box Blur) with different Kernel size', size=30)
pylab.show()

print('FIM')