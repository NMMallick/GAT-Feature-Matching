import numpy as np
import cv2 as cv

img = cv.imread('../imgs/ellie.jpg')
sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(img, None)

# Save SIFT descriptors to a .npy file
np.save('features/descriptors.npy', des)
print('ehllo')
