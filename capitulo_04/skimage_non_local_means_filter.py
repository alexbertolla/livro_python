import numpy as np
from skimage import data, img_as_float, img_as_ubyte, exposure, io, color
from skimage.io import imread
from skimage.exposure import cumulative_distribution
from skimage.restoration import denoise_bilateral, denoise_nl_means, estimate_sigma
from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio
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
denoise = denoise_nl_means(noisy, h=1.15 * sigma_est, fast_mode=False, **patch_wk)

#fat algorithm
denoise_fast = denoise_nl_means(noisy, h=0.8 * sigma_est, fast_mode=True, **patch_wk)

fig, axes = pylab.subplots(nrows=2, ncols=2, figsize=(15, 12), sharex=True, sharey=True)
plot_image_axes(noisy, axes[0, 0], 'noisy')
plot_image_axes(denoise, axes[0, 1], 'non-local means (slow)')
plot_image_axes(parrot, axes[1, 0], 'original (noise free)')
plot_image_axes(denoise_fast, axes[1, 1], 'non-local means (fast)')

#PSNR metrics values
#UserWarning: DEPRECATED: skimage.measure.compare_psnr has been moved to skimage.metrics.peak_signal_noise_ratio. It will be removed from skimage.measure in version 0.18.
psnr_noisy = peak_signal_noise_ratio(parrot, noisy)
psnr = peak_signal_noise_ratio(parrot, denoise.astype(np.float64))
psnr_fast = peak_signal_noise_ratio(parrot, denoise_fast.astype(np.float64))

print('PSNR (noisy) = {:0.2f}'. format((psnr_noisy)))
print('PSNR (slow) = {:0.2f}'. format((psnr)))
print('PSNR (fast) = {:0.2f}'. format((psnr_fast)))



pylab.show()

print('FIM')