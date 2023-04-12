import numpy as np
import cv2 as cv

## @brief Draw the relative features between two images
## @param gsImg Grayscale image used to generate the features
## @param kp Keypoints generated from the feature extractor
## param img
def drawImages(gs_img, kp, img):
    img = cv.drawKeypoints(gs_img, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite('../imgs/sift_keypoints.jpg', img)


## @brief Extract the keypoints and descriptors of an image
## @param img The image (in grayscale) to be passed through the detector
## @param ext The extractor use to extract keypoints and descriptors
def extractFeatures(img, ext):

    ## Create the feature extractor and find the descriptors of the img
    kp, des = ext.detectAndCompute(img, None)
    return (kp, des)


def computeEuclideanDistance(desc_a, desc_b):
    mat = []

    for i, d_a in enumerate(desc_a):
        a = np.array(d_a)
        for j,d_b in enumerate(desc_b):
            b = np.array(d_b)
            d = np.linalg.norm(b-a)
            if d == 0:
                mat.append([i,j])

    print(mat)

if __name__ == '__main__':

    img_descriptors = []

    ## Extract the image
    img = cv.imread('../imgs/ellie.jpg')

    ## Create the SIFT feature extractor
    gs_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()

    kp, des = extractFeatures(gs_img, sift)

    img_descriptors.append(des)
    img_descriptors.append(des)

    computeEuclideanDistance(img_descriptors[0], img_descriptors[1])
    drawImages(gs_img, kp, img)
