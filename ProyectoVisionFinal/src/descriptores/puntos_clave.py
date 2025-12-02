import cv2

def descriptor_SIFT(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def descriptor_ORB(img, n=500):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(n)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

def descriptor_AKAZE(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(gray, None)
    return keypoints, descriptors
