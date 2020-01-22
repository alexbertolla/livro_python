#DEVE-SE IMPORTAR O PACOTE opencv-contrib

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
import cv2

img = cv2.imread('../images/monalisa.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None) #DETECT SIFT KEYPOINTS

img = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Image', img)

kp, des = sift.detectAndCompute(gray, None) #COMPUTE THE SIFT DESCRIPTOR
print(des)

cv2.waitKey(0)
cv2.destroyAllWindows()


print('FIM')