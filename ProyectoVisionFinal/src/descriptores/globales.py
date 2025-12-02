import cv2
import numpy as np

##Histograma color
def histograma_color(img):
    hist = []
    for i in range(3):  # B, G, R
        h = cv2.calcHist([img], [i], None, [256], [0, 256])
        hist.extend(h.flatten())
    return np.array(hist)

## Histograma HOG
def HOG_descriptor(img):
    hog = cv2.HOGDescriptor()
    img_resized = cv2.resize(img, (128, 128))
    descriptor = hog.compute(img_resized)
    return descriptor.flatten()
