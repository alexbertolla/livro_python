#DEFEAULT IMPORTS
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from PIL.ImageChops import add, subtract, multiply, difference, screen
import PIL.ImageStat as stat
from skimage.io import imread, imsave, imshow, show, imread_collection, imshow_collection
from skimage import color, exposure, img_as_float, data, viewer
from skimage.transform import SimilarityTransform, warp, swirl
from skimage.util import invert, random_noise, montage
import matplotlib.image as mpimg
from matplotlib import pylab as pylab
from scipy.ndimage import affine_transform, zoom
from scipy import misc

#CHAPTER IMPORTS
from skimage.feature import hog
from skimage import exposure


image = color.rgb2gray(imread('../images/cameraman.jpg'))
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
print(image.shape, len(fd))

fig, (axes1, axes2) = pylab.subplots(1, 2, figsize=(9, 6), sharex=True, sharey=True)
axes1.axis('off'), axes1.imshow(image, cmap=pylab.cm.gray), axes1.set_title('Imput Image')

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

axes2.axis('off'), axes2.imshow(hog_image_rescaled, cmap=pylab.cm.gray), axes2.set_title('Histogram of Oriented Gradient')

pylab.show()

print('FIM')