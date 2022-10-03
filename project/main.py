"""
    Name: Athena Hernandez
    Date: September 29, 2022
    Description: This programs scans a paper :)
"""

# from pyimagesearch.transform import four_point_transform
# from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2 as cv
import imutils
from imutils.perspective import four_point_transform as fourPointTransform

def orderCorners(corners):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = corners.sum(axis = 1)
	rect[0] = corners[np.argmin(s)]
	rect[2] = corners[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(corners, axis = 1)
	rect[1] = corners[np.argmin(diff)]
	rect[3] = corners[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def warp(img, corners):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = orderCorners(corners)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv.getPerspectiveTransform(rect, dst)
	warped = cv.warpPerspective(img, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def findCnt(edged):
    cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:5]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        perimeter = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * perimeter, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            return approx

def main():
    # parse = argparse.ArgumentParser()
    # parse.add_argument('-i', '--image', required=True, help='path to image')
    # args = vars(parse.parse_args())

    # Image path given directly instead of console argument
    path = "images/practice.jpg"
    img = cv.imread(path) 
    ratio = img.shape[0] / 500.0
    orig = img.copy()
    width = img.shape[1] # number of columns
    height = img.shape[0] # number of rows
    img = imutils.resize(img, height = 500)

    # Resize image so functions well with OpenCV
    # img = cv.resize(img, (1300, 800))
    original_img = img.copy()

    # Grayscale image
    grayscaled = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("Grayscaled", grayscaled)
    cv.waitKey(0)

    # Blur image
    blurred = cv.GaussianBlur(grayscaled, (5, 5), 0)
    cv.imshow("Blurred", blurred)
    cv.waitKey(0)

    # Edge image
    edged = cv.Canny(blurred, 30, 50)
    cv.imshow("Edged", edged)
    cv.waitKey(0)

    # Find paper's contour
    cnt = findCnt(edged.copy())
    # show the contour (outline) of the piece of paper
    cv.drawContours(img, [cnt], -1, (0, 255, 0), 2)
    cv.imshow("Contours", img)
    cv.waitKey(0)

    # Warp image
    warped = warp(orig, cnt.reshape(4, 2) * ratio)
    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    # T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    # warped = (warped > T).astype("uint8") * 255
    # show the original and scanned images
    # cv.imshow("Original", imutils.resize(orig, height = 650))
    cv.imshow("Warped", cv.resize(warped, (width, height)))
    cv.waitKey(0)

    warpedAgain = fourPointTransform(orig, cnt.reshape(4, 2) * ratio)
    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warpedAgain = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    # T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    # warped = (warped > T).astype("uint8") * 255
    # show the original and scanned images
    # cv.imshow("Original", imutils.resize(orig, height = 650))
    cv.imshow("Warped Again", cv.resize(warpedAgain, (width, height)))
    cv.waitKey(0)

if __name__ == '__main__':
    main()

# Affine
# Live video feed; text isolation; see if text is vertical pytessaract
