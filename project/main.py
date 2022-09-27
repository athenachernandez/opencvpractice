import numpy as np
import cv2
import argparse



# parse = argparse.ArgumentParser()
# parse.add_argument('-i', '--image', required=True, help='path to image')
# args = vars(parse.parse_args())

path = "images/image.jpg"
img = cv2.imread(path)

# Resize image so functions well with OpenCV
img = cv2.resize(img, (1300, 800))
original_img = img.copy()

# STEP 1: Grayscale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray_img)
cv2.waitKey(0)

# STEP 2: Blur image
blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
cv2.imshow("Blur", blur_img)
cv2.waitKey(0)

# STEP 3: Edge detection
edge_img = cv2.Canny(blur_img, 30, 50)
cv2.imshow("Canny Edges", edge_img)
cv2.waitKey(0)

# cv2.destroyAllWindows()

# STEP 4: Find countours
img, contours = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
big = max(contours, key = cv2.contourArea)
# Find largest countour which is the page outline; search reverse order
# contours = sorted(contours, key = cv2.contourArea, reverse = True)

# Affine
# Live video feed; text isolation; see if text is vertical 

# for contour in contours:
#     cv2.arcLength(contour, True) # Tries to find closed shape
#     approx = cv2.approxPolyDP(contour,)

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True,
# help = "Path to the image")
# args = vars(ap.parse_args())

# img = cv2.imread("test.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.subtract(255,gray)
# ret,thresh = cv2.threshold(gray,5,255,cv2.THRESH_TOZERO)
# kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
# kernel2 = np.ones((3,3),np.uint8)
# erosion = cv2.erode(thresh,kernel2,iterations = 1)
# dilation = cv2.dilate(erosion,kernel1,iterations = 7)

