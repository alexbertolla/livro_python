import numpy as np
import numpy.fft as fp
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import img_as_float, img_as_uint, img_as_ubyte
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
im_cv = cv.cvtColor(cv.imread('../images/lena.jpg'), cv.COLOR_BGR2GRAY)
im_cv_b = cv.blur(im_cv, (5, 5))
im_cv_b = img_as_float(im_cv_b)

epsilon = 10**-6
freq = fp.fft2(im_cv_b)
freq_kernel = 1 / (epsilon + freq_kernel) #avoid division by zero
convolved = freq*freq_kernel

im_cv_r = fp.ifft2(convolved).real
im_cv_r = 255 * im_cv_r / np.max(im_cv)




########################################################################################################################

pylab.figure(figsize=(5, 5))
pylab.subplot(321), pylab.imshow(im, cmap='gray'), pylab.title('Original Image'), pylab.axis('off')
pylab.subplot(322), pylab.imshow(im_blur, cmap='gray'), pylab.title('Blurred Image'), pylab.axis('off')
pylab.subplot(323), pylab.imshow(im_restored, cmap='gray'), pylab.title('Restored Image'), pylab.axis('off')
pylab.subplot(324), pylab.imshow(im_cv_b, cmap='gray'), pylab.title('OPEN CV Image Blurred'), pylab.axis('off')
pylab.subplot(325), pylab.imshow(im_cv_r, cmap='gray'), pylab.title('OPEN CV Image Restored'), pylab.axis('off')

pylab.show()
print('FIM')