from __future__ import print_function
import argparse
import cv2

ap = argparse.ArgumentParser()
# Arguments required for running file
ap.add_argument("-i", "--image", required = True, help  = "path to file")
args = vars(ap.parse_args())

# imread() loads it off the disk and returns a NumPy array representing the image
image = cv2.imread(args["image"])

print(image)
# Examines dimensions of image
print("width: {} pixels".format(image.shape[1]))
print("height: {} pixels".format(image.shape[0]))
print("channels: {}".format(image.shape[2])) # colors

# Actual showing the image
cv2.imshow("Image", image)
cv2.waitKey(0)

# Name it
cv2.imwrite("newimage.jpg", image)