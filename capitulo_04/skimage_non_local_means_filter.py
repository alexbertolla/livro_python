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


def plot_image_axes(image, axes, title):
    axes.imshow(image)
    axes.axis('off')
    axes.set_title(title, size=20)


parrot = imread('../images/parrot.png')
parrot = img_as_float(parrot)

sigma = 0.25

noisy = parrot + sigma * np.random.standard_normal(parrot.shape)
noisy = np.clip(noisy, 0,1)

#estimate the noise standard deviation from the noise image
sigma_est = np.mean(estimate_sigma(noisy, multichannel=True))
print('estimated noise standard deviation = {}'.format(sigma_est))

#patch_distance 5x5 patches | patch_distance 13x13 search area
patch_wk = dict(patch_size=5, patch_distance=6, multichannel=True)

#slow algorithm


print('FIM')