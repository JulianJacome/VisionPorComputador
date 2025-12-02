import cv2
import numpy as np

def LBP(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = np.zeros_like(gray)

    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            center = gray[i, j]
            binary = []

            binary.append(1 if gray[i-1, j-1] >= center else 0)
            binary.append(1 if gray[i-1, j] >= center else 0)
            binary.append(1 if gray[i-1, j+1] >= center else 0)
            binary.append(1 if gray[i, j+1] >= center else 0)
            binary.append(1 if gray[i+1, j+1] >= center else 0)
            binary.append(1 if gray[i+1, j] >= center else 0)
            binary.append(1 if gray[i+1, j-1] >= center else 0)
            binary.append(1 if gray[i, j-1] >= center else 0)

            lbp[i, j] = int("".join(str(x) for x in binary), 2)

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))

    return hist
