import numpy as np
import matplotlib.pyplot as pylab
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from skimage.io import imread
from skimage import img_as_float


pepper = imread('../images/pimentao.jpg')

# Convert to floats instead of the default 8 bits integer coding, so that
# pylab.imshow behaves works well in float data (need ti be in the range [0-1]
pepper = img_as_float(pepper)

# Load Image and tranform to a 3D numpy array
w, h, d = original_shape = tuple(pepper.shape)

assert d == 3
image_array = np.reshape(pepper, (w * h, d))

def recreate_image(codebook, labels, w, h):
    """ Recreate the (compressed) image from the code book & lables """
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    #print(image)
    #exit()
    return image

#Display all results, alongside original image
pylab.figure(1), pylab.clf()
ax = pylab.axes([0, 0, 1, 1])
pylab.axis('off'), pylab.title('Orignal image (%d colors)' %(len(np.unique(pepper)))), pylab.imshow(pepper)
pylab.show()

pylab.figure(2, figsize=(10, 10)), pylab.clf()
i = 1
for k in [256, 128, 64, 32, 16, 4, 2, 1]:
    # Run Kmeans on random sample of 1000 pixels from the image
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(image_array_sample)

    # Predicting color indices on the full imagen (K-means)
    labels = kmeans.predict(image_array)
    pylab.subplot(4, 4, i), pylab.axis('off'), pylab.title('Quantized iamge (' + str(k) + ' colors, K-means)')
    pylab.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
    i += 1
pylab.show()

pylab.figure(3, figsize=(10, 10)), pylab.clf()
i = 1
for k in [256, 128, 64, 32, 16, 4, 2, 1]:
    codebook_random = shuffle(image_array, random_state=0)[:k + 1]

    # Predicting color indices on the full imagen (random)
    labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)
    pylab.subplot(4, 4, i), pylab.axis('off'), pylab.title('Quantized iamge (' + str(k) + ' colors, Random)')
    pylab.imshow(recreate_image(codebook_random, labels_random, w, h))
    print(i)
    i += 1
pylab.show()


print('FIM')