import numpy as np
import numpy.fft as fp
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import img_as_float, img_as_uint, img_as_ubyte
from skimage.filters import gaussian
from scipy import signal
from matplotlib import pylab
import cv2 as cv

im = rgb2gray(imread('../images/lena.jpg'))
gauss_kernel = np.outer(signal.gaussian(im.shape[0], 3), signal.gaussian(im.shape[1], 3))

freq = fp.fft2(im)
freq_kernel = fp.fft2(fp.ifftshift(gauss_kernel)) #this is our H
convolved = freq*freq_kernel #by convolution theorem
im_blur = fp.ifft2(convolved).real
im_blur = 255 * im_blur / np.max(im_blur) #normalize

epsilon = 10**-6
freq = fp.fft2(im_blur)
freq_kernel = 1 / (epsilon + freq_kernel) #avoid division by zero
convolved = freq*freq_kernel

im_restored = fp.ifft2(convolved).real
im_restored = 255 * im_restored / np.max(im_restored)

########################################################################################################################

#im2 = imread('../images/lena_template.png')
im2 = imread('../images/lena_filtro_media_7_7.jpg')

im2 = gaussian(im2, 5)
gauss_kernel2 = np.outer(signal.gaussian(im2.shape[0], 3), signal.gaussian(im2.shape[1], 3))
print(gauss_kernel2)

freq2 = fp.fft2(im2)
freq_kernel2 = fp.fft2(fp.ifftshift(im2)) #this is our H
convolved2 = freq2*freq_kernel2 #by convolution theorem
im_blur2 = fp.ifft2(convolved2).real
im_blur2 = 255 * im_blur2 / np.max(im_blur2) #normalize

epsilon = 10**-6
freq3 = fp.fft2(im_blur2)
freq_kernel3 = 1 / (epsilon + freq_kernel2) #avoid division by zero
convolved3 = freq3*freq_kernel3

im_restored2 = fp.ifft2(convolved3).real
im_restored2 = 255 * im_restored2 / np.max(im_restored2)



########################################################################################################################

pylab.figure(figsize=(5, 5))
pylab.subplot(321), pylab.imshow(im, cmap='gray'), pylab.title('Original Image'), pylab.axis('off')
pylab.subplot(322), pylab.imshow(im_blur, cmap='gray'), pylab.title('Blurred Image'), pylab.axis('off')
pylab.subplot(323), pylab.imshow(im_restored, cmap='gray'), pylab.title('Restored Image'), pylab.axis('off')
pylab.subplot(325), pylab.imshow(im2, cmap='gray'), pylab.title('Blurred Image 2'), pylab.axis('off')
pylab.subplot(326), pylab.imshow(gauss_kernel, cmap='gray'), pylab.title('Restored Image 2'), pylab.axis('off')

pylab.show()
print('FIM')