from __future__ import print_function
import numpy as np
import argparse
import imutils
import glob
import cv2
import os

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


# function to align the thermal and visible image, it returns the homography matrix
def alignImages(im1, im2, filename):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    matches = list(matches)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite(os.path.join('./registration/', filename), imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    # im1Reg = im1 @ h  # I added it, but it doesn't do anything

    return im1Reg, h


# construct the argument parser and parse the arguments
# run the file with python registration.py --image filename
ap = argparse.ArgumentParser()
#ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--image", required=True,
                help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize",
                help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())

# put the thermal image in a folder named thermal and the visible image in a folder named visible with the same name
# load the image image, convert it to grayscale, and detect edges
template = cv2.imread('thermal/' + args["image"] + '.jpg')
template = cv2.resize(template, (640, 480))
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
chessboard_size = (5, 7)
# Detect chessboard corners
ret, corners = cv2.findChessboardCorners(template, chessboard_size, None)
mask = np.zeros_like(template)
cv2.fillConvexPoly(mask, cv2.convexHull(corners), 255)
canny_edges = cv2.Canny(template, 50, 150)
template = cv2.bitwise_and(canny_edges, mask)
# template = cv2.Canny(template, 40, 50)
(tH, tW) = template.shape[:2]
cv2.imshow("Template", template)
cv2.waitKey(0)


# loop over the images to find the template in

# load the image, convert it to grayscale, and initialize the
# bookkeeping variable to keep track of the matched region
image = cv2.imread('visible/' + args["image"] + '.jpg')
# image = cv2.resize(image, (640, 480))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
found = None
# loop over the scales of the image
for scale in np.linspace(1.0, 2.0, 20)[::-1]:
    # resize the image according to the scale, and keep track
    # of the ratio of the resizing
    resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
    r = gray.shape[1] / float(resized.shape[1])

    # if the resized image is smaller than the template, then break
    # from the loop
    if resized.shape[0] < tH or resized.shape[1] < tW:
        break

    # detect edges in the resized, grayscale image and apply template
    # matching to find the template in the image
    edged = cv2.Canny(resized, 40, 50)
    result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

    # check to see if the iteration should be visualized
    if args.get("visualize", False):
        # draw a bounding box around the detected region
        clone = np.dstack([edged, edged, edged])
        cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                      (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
        cv2.imshow("Visualize", clone)
        cv2.waitKey(0)

    # if we have found a new maximum correlation value, then update
    # the bookkeeping variable
    if found is None or maxVal > found[0]:
        found = (maxVal, maxLoc, r)

# unpack the bookkeeping variable and compute the (x, y) coordinates
# of the bounding box based on the resized ratio
(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

# draw a bounding box around the detected result and display the image
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
crop_img = image[startY:endY, startX:endX]
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.imshow("Image", crop_img)
cv2.waitKey(0)

name = "thermal/" + args["image"] + '.jpg'
thermal_image = cv2.imread(name, cv2.IMREAD_COLOR)

# cropping out the matched part of the thermal image
crop_img = cv2.resize(crop_img, (thermal_image.shape[1], thermal_image.shape[0]))

# cropped image will be saved in a folder named output
cv2.imwrite(os.path.join('./output/', args["image"] + '.jpg'), crop_img)

# both images are concatenated and saved in a folder named results
final = np.concatenate((crop_img, thermal_image), axis=1)
cv2.imwrite(os.path.join('./results/', args["image"] + '.jpg'), final)

cv2.waitKey(0)
# Registration
# Read reference image
refFilename = "thermal/" + args["image"] + '.jpg'
print("Reading reference image : ", refFilename)
imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

# Read image to be aligned
imFilename = "output/" + args["image"] + '.jpg'
print("Reading image to align : ", imFilename);
im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
file_name = args["image"] + '.jpg'
imReg, h = alignImages(im, imReference, file_name)
print("Estimated homography : \n", h)

cv2.imshow("Image 0", imReg)
cv2.waitKey(0)
