import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob
import json

path = 'Imgs'

# Image resolution for processing
img_width = 640
img_height = 480
image_size = (img_width, img_height)

# inner size of chessboard
rows = 5
columns = 6
square_size = 0.026  # 0.026 meters

CHECKERBOARD = (rows, columns)

# Visualization options
drawCorners = True
visual = True

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 1, 3), np.float64)
objp[:, 0, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objp = objp * square_size  # Create real world coords. Use your metric.

objpoints = []  # 3d point in real world space
imgpointsLeft = []  # 2d points in image plane.
imgpointsRight = []  # 2d points in image plane.

image_dir = path

images_array_left = glob.glob('Imgs/*vis*.jpg')
images_array_right = glob.glob('Imgs/*thermal*.jpg')
number_of_images = len(images_array_left)

for i in range(1, 64):
    # read image
    image_left = cv2.imread(images_array_left[i - 1])
    image_right = cv2.imread(images_array_right[i - 1])
    image_right = cv2.resize(image_right, (img_width, img_height))

    gray_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
    # gray_right = cv2.resize(gray_right, (img_width, img_height))
    # gray_left = cv2.GaussianBlur(gray_left, (5, 5), 0)
    Y, X, channels = image_left.shape
    #
    # threshold_value = 180
    # max_value = 255
    # _, binary_image = cv2.threshold(gray_left, threshold_value, max_value, cv2.THRESH_BINARY)
    #
    # cv2.imshow('Threshold', binary_image)
    # cv2.waitKey(0)

    # gray_small_left = cv2.resize (gray_left, (img_width,img_height), interpolation = cv2.INTER_AREA)
    # gray_small_right = cv2.resize (gray_right, (img_width,img_height), interpolation = cv2.INTER_AREA)

    # Find the chessboard corners
    retL, cornersL = cv2.findChessboardCornersSB(gray_left, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    retR, cornersR = cv2.findChessboardCornersSB(gray_right, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # Draw images with corners found
    if (drawCorners):
        cv2.drawChessboardCorners(image_left, CHECKERBOARD, cornersL, retL)
        cv2.imshow('Corners LEFT', image_left)
        cv2.drawChessboardCorners(image_right, CHECKERBOARD, cornersR, retR)
        cv2.imshow('Corners RIGHT', image_right)

        # accepted = False
        # key = cv2.waitKey(0)
        # if key == ord("q"):
        #     exit(0)
        # elif key == ord("a"):
        #     accepted = True

        accepted = True
        if i in {2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 26, 29, 32, 33, 36, 38, 40, 41, 48, 49, 
                 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 65, 67, 68}:
            accepted = False

        # if i == 3:
        #     cv2.waitKey(0)
        #     withChess = image_right

    SayMore = False  # Should we print additional debug info?

    # Refine corners and add to array for processing
    if retL and retR and accepted:

        objpoints.append(objp)

        cv2.cornerSubPix(gray_left, cornersL, (3, 3), (-1, -1), subpix_criteria)
        cornersL = cornersL[::-1]
        imgpointsLeft.append(cornersL)

        # objpointsRight.append(objp)

        cv2.cornerSubPix(gray_right, cornersR, (3, 3), (-1, -1), subpix_criteria)
        imgpointsRight.append(cornersR)
    else:
        print("Chessboard couldn't detected. Image pair: ", i)
        continue

## CALIBRATION ONE CAMERA

N_OK = len(objpoints)
# DIM = (img_width, img_height)
K_left = np.zeros((3, 3))
D_left = np.zeros((4, 1))
rvecs_left = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs_left = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

print(objpoints)
print(len(objpoints))
# Single camera calibration (undistortion)
rms, K_left, D_left, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpointsLeft,
        image_size,
        K_left,
        D_left,
        rvecs_left,
        tvecs_left,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
# Let's rectify our results
map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(K_left, D_left, np.eye(3), K_left, image_size, cv2.CV_16SC2)

data = {"ret": rms, "mtx": K_left, "dist": D_left, "rvecs": None, "tvecs": None, "total_error": None}
with open('calibParams.pkl', 'wb') as file:
    pickle.dump(data, file)

K_right = np.zeros((3, 3))
D_right = np.zeros((4, 1))
rvecs_right = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs_right = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

# Single camera calibration (undistortion)
rms, K_right, D_right, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpointsRight,
        image_size,
        K_right,
        D_right,
        rvecs_right,
        tvecs_right,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
# Let's rectify our results
map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, np.eye(3), K_right, image_size,
                                                             cv2.CV_16SC2)

print("Found " + str(N_OK) + " valid images for calibration")
print("DIM =" + str(image_size))

print("Left image")
print("K = np.array(" + str(K_left.tolist()) + ")")
print("D = np.array(" + str(D_left.tolist()) + ")")

print("Right image")
print("K = np.array(" + str(K_right.tolist()) + ")")
print("D = np.array(" + str(D_right.tolist()) + ")")

# SHOW SINGLE CAM CALIBRATION UNDISTORTED

image_left = cv2.imread(image_dir + "/vis_photo_1730746556.jpg")
image_right = cv2.imread("Imgs5/thermal_photo_1730746556.jpg")
image_right = cv2.resize(image_right, (img_width, img_height))

gray_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

undistorted_left = cv2.remap(image_left, map1_left, map2_left, interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
undistorted_right = cv2.remap(image_right, map1_right, map2_right, interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)

if visual:
    cv2.imshow('Left UNDISTORTED', undistorted_left)
    cv2.imshow('Right UNDISTORTED', undistorted_right)
    cv2.waitKey(0)

# STEREO CALIBRATION
imgpointsLeft = np.asarray(imgpointsLeft, dtype=np.float64)
imgpointsRight = np.asarray(imgpointsRight, dtype=np.float64)

(RMS, _, _, _, _, rotationMatrix, translationVector) = cv2.fisheye.stereoCalibrate(
    objpoints, imgpointsLeft, imgpointsRight,
    K_left, D_left,
    K_right, D_right,
    image_size, None, None,
    cv2.CALIB_FIX_INTRINSIC,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))

# Print RMS result (for calibration quality estimation)
print("####   RMS is ", RMS, " ####")

print("Rectifying cameras...")
R1 = np.zeros([3, 3])
R2 = np.zeros([3, 3])
P1 = np.zeros([3, 4])
P2 = np.zeros([3, 4])
Q = np.zeros([4, 4])

# Rectify calibration results
(leftRectification, rightRectification, leftProjection, rightProjection,
 dispartityToDepthMap) = cv2.fisheye.stereoRectify(
    K_left, D_left,
    K_right, D_right,
    image_size,
    rotationMatrix, translationVector,
    0, R2, P1, P2, Q,
    cv2.CALIB_ZERO_DISPARITY, (0, 0), 0, 0)

map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
    K_left, D_left, leftRectification,
    leftProjection, image_size, cv2.CV_16SC2)

map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
    K_right, D_right, rightRectification,
    rightProjection, image_size, cv2.CV_16SC2)

imgL = cv2.remap(image_left, map1_left, map2_left, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
imgR = cv2.remap(image_right, map1_right, map2_right, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

two = cv2.hconcat([imgL, imgR])
cv2.imshow("Con", two)
cv2.waitKey(0)

if visual:
    print("Display calibrated stereo images")

    cv2.imshow('Left STEREO CALIBRATED', imgL)
    cv2.imshow('Right STEREO CALIBRATED', imgR)
    cv2.waitKey(0)

XX, YY = (img_width, img_height)
visRectify = np.zeros((YY, XX * 2, 3), np.uint8)  # Create a new image with double width
visRectify[:, 0:XX, :] = imgL  # Assign the left image
visRectify[:, XX:XX * 2, :] = imgR  # Assign the right image

# Draw horizontal lines
for y in range(0, YY, 10):
    cv2.line(visRectify, (0, y), (XX * 2, y), (255, 0, 0))

visRectifyV = np.zeros((YY * 2, XX, 3), np.uint8)  # Create a new image with double height
visRectifyV[0:YY, :, :] = imgL  # Assign the left image
visRectifyV[YY:YY * 2, :, :] = imgR  # Assign the right image

# Draw vertical lines
for x in range(0, XX, 30):
    cv2.line(visRectifyV, (x, 0), (x, YY * 2), (255, 0, 0))

visRectifyV = cv2.resize(visRectifyV, (320, 480))

if (visual):
    cv2.imshow('visRectify', visRectifyV)  # Visualization
    cv2.waitKey(0)

# FOV pairing
ptsL = np.squeeze(cornersL)
ptsL = ptsL[[0, 4, -4, -1], :].astype(np.float32)
ptsR = np.squeeze(cornersR)
ptsR = ptsR[[0, 4, -4, -1], :].astype(np.float32)
M = cv2.getPerspectiveTransform(ptsL, ptsR)
M_list = M.tolist()

calibData = {"M": M_list}

with open("M.json", "w") as f:
    json.dump(calibData, f)

imgR_lin_warp = cv2.warpPerspective(imgR, M, (640, 480))
cv2.imshow("warp", imgR_lin_warp)
cv2.waitKey(0)
cv2.imwrite("imgR_lin_warp.jpg", imgR_lin_warp)

imgL_lin_warp = cv2.warpPerspective(imgL, M, (640, 480))
cv2.imshow("warp", imgL_lin_warp)
cv2.waitKey(0)
cv2.imwrite("imgL_lin_warp.jpg", imgL_lin_warp)

x1, x2, y1, y2 = 130, 340, 210, 340
pix_shift = -10
imgL = imgL_lin_warp[y1:y2, x1:x2]
imgR = imgR_lin_warp[y1:y2, (x1 + pix_shift):(x2 + pix_shift)]

new_width = x2 - x1
new_height = y2 - y1

XX, YY = (new_width, new_height)
visRectify = np.zeros((YY, XX * 2, 3), np.uint8)  # Create a new image with double width
visRectify[:, 0:XX, :] = imgL  # Assign the left image
visRectify[:, XX:XX * 2, :] = imgR  # Assign the right image

# Draw horizontal lines
for y in range(0, YY, 10):
    cv2.line(visRectify, (0, y), (XX * 2, y), (255, 0, 0))

visRectifyV = np.zeros((YY * 2, XX, 3), np.uint8)  # Create a new image with double height
visRectifyV[0:YY, :, :] = imgL  # Assign the left image
visRectifyV[YY:YY * 2, :, :] = imgR  # Assign the right image

# Draw vertical lines
for x in range(0, XX, 30):
    cv2.line(visRectifyV, (x, 0), (x, YY * 2), (255, 0, 0))

visRectifyV = cv2.resize(visRectifyV, (320, 480))

if (visual):
    cv2.imshow('visRectifyH', visRectify)  # Visualization
    cv2.waitKey(0)
    cv2.imshow('visRectifyV', visRectifyV)
    cv2.waitKey(0)

