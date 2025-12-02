import cv2
import numpy as np

def hu_moments(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments)
    
    # Escalar logarítmicamente para mayor estabilidad
    for i in range(0, 7):
        hu[i] = -1 * np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-10)
    
    return hu.flatten()
