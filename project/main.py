"""
    Name: Athena Hernandez
    Date: September 29, 2022
    Description: This programs scans a paper, reads text, and scans QR codes :)
"""

import numpy as np
import cv2 as cv
import imutils
from imutils.perspective import four_point_transform as fourPointTransform
import pytesseract
from pytesseract import Output
from colorama import Fore

def orderCorners(corners):
    """ Obtains a consistent order of points and unpacks them individually"""
    # List of coordinates starting at top-left going clockwise
    rect = np.zeros((4, 2), dtype = "float32")        

    s = corners.sum(axis = 1)                                           # Top-left = smallest sum; bottom-right = largest sum
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]

    # Compute other points
    diff = np.diff(corners, axis = 1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
	# return the ordered coordinates
    return rect                                                         # Ordered coordinates

def warp(img, corners):
    """ Aligns image to aerial view"""
    rect = orderCorners(corners)
    (tl, tr, br, bl) = rect
     
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))       # Distance between bottom right and left
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))       # Distance between top right and left
    maxWidth = max(int(widthA), int(widthB))                                # Finds largest edge to use as base to warp
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))      # Same idea for height
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Create points for warped image
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # Compute perspective transform matrix then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped                                                           # Warped image

def largestCnt(edged):
    """ Finds the largest rectangular contour """
    cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:5]   # Largest to smallest
    
    for c in cnts:                                                  # Loop over contours
        perimeter = cv.arcLength(c, True)                   
        approx = cv.approxPolyDP(c, 0.02 * perimeter, True)         # Approximates shape
        if len(approx) == 4:                                        # Paper has 4 points
            return approx

def qrScanner(filename):
    """ Uses QRCodeDetector and simple geometry to generate a link from a QR Code """
    img = cv.imread(f"images/{filename}")
    qrCodeDetector = cv.QRCodeDetector()
    decodedText, points, _ = qrCodeDetector.detectAndDecode(img)

    if points is not None:
        for i in range(len(points)):
            nextPointIndex = (i+1) % len(points)
            # 📌 Chapter 5.1: Lines and Rectangles 📌 
            cv.line(img, (int(points[i][0][0]), int(points[i][0][1])), (int(points[nextPointIndex][0][0]), int(points[nextPointIndex][0][1])), (255, 0, 0), 5)
            print(f"The QR code you selected brings you to this link: {Fore.BLUE}{decodedText}{Fore.BLACK}.")
            cv.imshow("Image", img)
            cv.waitKey(0)
    else:
        print("QR code not detected :(")

def textDetector(filename):
    """ Uses Pytesseract to translate image to text and detect where text is on screen """
    adaptiveThreshold = documentScanner(filename)
    pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.2.0/bin/tesseract'
    extractedText = pytesseract.image_to_string(adaptiveThreshold)
    
    print(f"{Fore.GREEN}\n{extractedText.strip()}\n")

    adaptiveThreshold = cv.cvtColor(adaptiveThreshold, cv.COLOR_GRAY2BGR)
    data = pytesseract.image_to_data(adaptiveThreshold, output_type = Output.DICT)
    numBoxes = len(data['level'])
    for i in range(numBoxes):
        # 📌 Chapter 4.3: Accessing and Manipulating Pixels 📌
        x = data['left'][i]
        y = data['top'][i]
        width = data['width'][i]
        height = data['height'][i]
        # 📌 Chapter 5.1: Lines and Rectangles 📌 
        cv.rectangle(adaptiveThreshold, (x, y), (x + width, y + height), (168, 122, 225), 2)

    cv.imshow('Boxed Text', adaptiveThreshold)
    cv.waitKey(0)

def display(title, img):
    """ Shortens display notation for images """
    cv.imshow(title, img)
    cv.waitKey(0)

def documentScanner(filename):
    """ Displays step-by-step how to create a scanned document and returns scan """
    img = cv.imread(f"images/{filename}") 
    og = img.copy()
    width = img.shape[1]                                    # Number of columns
    height = img.shape[0]                                   # Number of rows
    ratio = height / 500.0

    # 📌 Chapter 6.1.3: Resizing 📌
    img = imutils.resize(img, height = 500)                 # Resizing image is a standard practice for better results

    # 📌 Chapter 6.6: Color Spaces 📌
    grayscaled = cv.cvtColor(img, cv.COLOR_BGR2GRAY)        # Grayscale image to simplify calculations and remove redundancies
    display("Grayscaled", grayscaled)

    # 📌 Chapter 8.2: Gaussian Blurring 📌
    blurred = cv.GaussianBlur(grayscaled, (5, 5), 0)
    display("Blurred", blurred)

    # 📌 Chapter 10.2: Canny Edge Detection 📌
    edged = cv.Canny(blurred, 30, 50)
    display("Edged", edged)

    # 📌 Chapter 11.1 & 11.2: Contours  📌
    cnt = largestCnt(edged.copy())                          # Finds largest contour
    cv.drawContours(img, [cnt], -1, (168, 122, 225), 2)     # Outlines contour 
    display("Largest contour", img)

    # Warp image to aerial view
    warped = warp(og, cnt.reshape(4, 2) * ratio)
    warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    display("Warped", cv.resize(warped, (width, height)))

    # warpedAgain = fourPointTransform(orig, cnt.reshape(4, 2) * ratio)
    # warpedAgain = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    # display("Warped Again", cv.resize(warpedAgain, (width, height)))

    # 📌 Chapter 9.2: Adaptive Thresholding 📌
    adaptiveThreshold = cv.adaptiveThreshold(warped, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, 2)
    # 📌 Chapter 6.3: Bitwise Operations 📌
    adaptiveThreshold = cv.bitwise_not(adaptiveThreshold)
    # 📌 Chapter 8.3: Median Blurring 📌
    adaptiveThreshold = cv.medianBlur(adaptiveThreshold, 3) # Removes s&p noise
    display("Adaptive Threshold", adaptiveThreshold)        # Adds grainy scan-feel

    return adaptiveThreshold


def main():
    
    print(f"\nHey there! My OpenCV project can broken down into 3 parts. Press any number besides 1, 2, or 3 to exit.")
    
    inputText = "I'm going to assume you're choosing an image inside the \'images\' folder, however, no need to type in the whole path. Enter the image file name of your choice here: "
    while True:
        print(f"{Fore.RED}\n\t(1) Document scanner AKA Walmart Scannable\n\t{Fore.GREEN}(2) Text detector\n\t{Fore.BLUE}(3) QR scanner")
        selected = int(input(f"{Fore.BLACK}\nWhich would you like to try out? Enter {Fore.RED}1{Fore.BLACK}, {Fore.GREEN}2{Fore.BLACK}, or {Fore.BLUE}3{Fore.BLACK}: "))
        if selected == 1:
            print(f"\n🧾🧾🧾 Welcome to my {Fore.RED}document scanner{Fore.BLACK} AKA a Walmart version of Evernote's Scannable! 🧾🧾🧾\n")
            filename = input(inputText)
            documentScanner(filename)
        elif selected == 2:
            print(f"\n🔎🔎🔎 Welcome to my {Fore.GREEN}text detector{Fore.BLACK}! 🔍🔍🔍\n")
            filename = input(inputText)
            textDetector(filename)
        elif selected == 3:
            print(f"\n🔗🔗🔗 Welcome to my {Fore.BLUE}QR detector{Fore.BLACK}! 🔗🔗🔗\n")
            filename = input(inputText)
            qrScanner(filename)
        else:
            print("\nThanks for checking me out! ✌️ 😎\n")
            break
        cv.destroyAllWindows()
        print(f"{Fore.BLACK}Cool! Try again or press any number besides 1, 2, or 3 to quit.")

if __name__ == '__main__':
    main()
