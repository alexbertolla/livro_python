import numpy as np
#import numpy.fft as fp
from scipy.fftpack import fft2, fftn, ifft2, ifftn, fftshift, ifftshift
from PIL import Image
from matplotlib import pylab


im = np.array(Image.open('../images/rhino.jpg'))

freq = fftn(im)
(w, h, d) = freq.shape
half_w, half_h = int(w/2), int(h/2)

freq1 = np.copy(freq)
freq2 = fftshift(freq1)

original_spectrum = (20 * np.log10(0.1 + freq2)).astype(int)

#apply High pass filter

freq2[half_w-10:half_w+11, half_h-10:half_h+11] = 0 #select all but the first 20x20 low frequencies
high_pass_spectrum = (20 * np.log10(0.1 + freq2)).astype(int)

im1 = np.clip(ifftn(ifftshift(freq2)).real, 0, 255) #clip pixel values after IFFT


pylab.figure(figsize=(10, 5))
pylab.subplot(2, 2, 1), pylab.imshow(im), pylab.axis('off'), pylab.title('Rhino Original Image')
pylab.subplot(2, 2, 2), pylab.imshow(original_spectrum), pylab.title('Rhino Spectrum')
pylab.subplot(2, 2, 3), pylab.imshow(im1), pylab.axis('off'), pylab.title('Rhino High Pass Filtered')
pylab.subplot(2, 2, 4), pylab.imshow(high_pass_spectrum), pylab.title('Rhino High Pass Spectrum')


pylab.show()


print('FIM')