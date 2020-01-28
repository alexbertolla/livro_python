import cv2
import matplotlib.pylab as pylab

img = cv2.imread('../images/caminhando.jpg')

#CREATE HOG DESCRIPTOR USING DEFAULT PEOPLE DETECTOR
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


#RUN DETECTION, USING A SPATIAL STRIDE OF 4 PIXELS (HORIZONTAL AND VERTICAL), A SCALE STRIDE OF 1.02, AND ZERO GROUPING
#OF RECTANGLES (TO DEMONTRATE THAT HOG WILL DETECT AT  POTENTIALLY MULTIPLE PLACES IN THE SCALE PYRAMID)

(foundBoundingBoxes, weights) = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.02, finalThreshold=0)

print('Numeber of bouNdingBoxes = ', len(foundBoundingBoxes)) #NUMBER OF BOUNDINGBOXES

#COPY THE ORIGINAL IMAGE TO DRAW BOUNDING BOXES ON IT FOR NOW, AS WE WILL USE IT LATER

imgWithRawBoxes = img.copy()
for (hx, hy, hw, hh) in foundBoundingBoxes:
    cv2.rectangle(imgWithRawBoxes, (hx, hy), (hx+hw, hy+hh), (0, 0, 255), 1)

pylab.figure(figsize=(5, 3))
imgWithRawBoxes = cv2.cvtColor(imgWithRawBoxes, cv2.COLOR_BGR2GRAY)
pylab.gray(), pylab.imshow(imgWithRawBoxes, aspect='auto'), pylab.axis('off'), pylab.show()


print('FIM')