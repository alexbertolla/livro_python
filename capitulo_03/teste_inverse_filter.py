import numpy as np
import numpy.fft as fp
from skimage.io import imread
from scipy.fftpack import fft2, fftn, ifft2, ifftn, fftshift, ifftshift
from skimage.color import rgb2gray
from skimage import img_as_float, img_as_uint, img_as_ubyte
from skimage.filters import gaussian
from scipy import signal
from matplotlib import pylab

original_image = rgb2gray(imread('../images/blurred_cameraman.png'))
freq = fft2(original_image)


epsilon = 10**-6
gauss_kernel = np.outer(signal.gaussian(original_image.shape[0], 3), signal.gaussian(original_image.shape[1], 3))
print(gauss_kernel)
freq_kernel = fft2(ifftshift(gauss_kernel)) #this is our H
freq_kernel = 1 / (epsilon + freq_kernel) #avoid division by zero
convolved = freq*freq_kernel

im_restored = ifft2(convolved).real
im_restored = 255 * im_restored / np.max(im_restored)

pylab.figure(figsize=(5, 5))
pylab.subplot(121), pylab.imshow(original_image, cmap='gray'), pylab.title('Original Image Blurred'), pylab.axis('off')
pylab.subplot(122), pylab.imshow(gauss_kernel, cmap='gray'), pylab.title('Restored Image Blurred'), pylab.axis('off')


pylab.show()
print('FIM')