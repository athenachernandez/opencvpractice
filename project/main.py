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
    path = "images/image.jpg"
    img = cv.imread(path) 

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
    cnt = findCnt(edged)
    # show the contour (outline) of the piece of paper
    cv.drawContours(img, [cnt], -1, (0, 255, 0), 2)
    cv.imshow("Outline", img)
    cv.waitKey(0)

    # Warp image

    cv.destroyAllWindows()

    # warped = four_point_transform(img, screenCnt.reshape(4, 2))
    # convert the warped image to grayscale
    warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    cv.imshow("Scanned", cv.resize(warped, (600, 800)))
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()

# Affine
# Live video feed; text isolation; see if text is vertical 
