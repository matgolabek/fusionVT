import numpy as np
import cv2
import pickle
import json

visual = True
img_width, img_height = 640, 480

image_left = cv2.imread("Imgs/vis_photo_1730746547.jpg")
image_right = cv2.imread("Imgs/thermal_photo_1730746547.jpg")
image_right = cv2.resize(image_right, (img_width, img_height))

with open("calibStereoParams.pkl", "rb") as f:
    calibData = pickle.load(f)

RMS = calibData["RMS"]
rotationMatrix = calibData["rotMtx"]
translationVector = calibData["transVec"]
objpoints = calibData["objpoints"]
imgpointsLeft = calibData["imgpointsVis"]
imgpointsRight = calibData["imgpointsThermal"]
K_left = calibData["KVis"]
K_right = calibData["KThermal"]
D_left = calibData["DVis"]
D_right = calibData["DThermal"]
image_size = calibData["imgSize"]
M = calibData["M"]

print(K_right)
print(D_right)


print("####   RMS is ", RMS, " ####")

print("Rectifying cameras...")
R1 = np.zeros([3, 3])
R2 = np.zeros([3, 3])
P1 = np.zeros([3, 4])
P2 = np.zeros([3, 4])
Q = np.zeros([4, 4])

FOV = 0.  # powinno być ~0.6!!

# Rectify calibration results
(leftRectification, rightRectification, leftProjection, rightProjection,
 dispartityToDepthMap) = cv2.fisheye.stereoRectify(
    K_left, D_left,
    K_right, D_right,
    image_size,
    rotationMatrix, translationVector,
    0, R2, P1, P2, Q,
    cv2.CALIB_ZERO_DISPARITY, (0, 0), 0, FOV)

map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
    K_left, D_left, leftRectification,
    leftProjection, image_size, cv2.CV_16SC2)

map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
    K_right, D_right, rightRectification,
    rightProjection, image_size, cv2.CV_16SC2)

map1_left_list = map1_left.tolist()
map2_left_list = map2_left.tolist()
map1_right_list = map1_right.tolist()
map2_right_list = map2_right.tolist()

calibDataJSON = {
    "map1_vis": map1_left_list,
    "map2_vis": map2_left_list,
    "map1_thermal": map1_right_list,
    "map2_thermal": map2_right_list
}

print(map1_left.dtype, map2_left.dtype)
with open("calibStereo.json", "w") as f:
    json.dump(calibDataJSON, f)
imgL = cv2.remap(image_left, map1_left, map2_left, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
imgR = cv2.remap(image_right, map1_right, map2_right, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

imgR_lin_warp = cv2.warpPerspective(imgR, M, (640, 480))
imgL_lin_warp = cv2.warpPerspective(imgL, M, (640, 480))


cv2.imwrite("imgL.jpg", imgL)
cv2.imwrite("imgR.jpg", imgR)
# Shift left-right

x1, x2, y1, y2 = 130, 340, 210, 340
pix_shift = 0 # -10
imgL = imgL_lin_warp[y1:y2, x1:x2]
imgR = imgR_lin_warp[y1:y2, (x1 + pix_shift):(x2 + pix_shift)]

new_width = x2 - x1
new_height = y2 - y1

if (visual):
    print("Display calibrated stereo images")

    cv2.imshow('Left STEREO CALIBRATED', imgL)
    cv2.imshow('Right STEREO CALIBRATED', imgR)
    cv2.waitKey(0)

XX, YY = (new_width, new_height)
visRectify = np.zeros((YY, XX * 2, 3), np.uint8)  # utworzenie nowego obrazka o szerokosci x2
visRectify[:, 0:XX, :] = imgL  # przypisanie obrazka lewego
visRectify[:, XX:XX * 2, :] = imgR  # przypisanie obrazka prawego

# Wyrysowanie poziomych linii
for y in range(0, YY, 10):
    cv2.line(visRectify, (0, y), (XX * 2, y), (255, 0, 0))

visRectifyV = np.zeros((YY * 2, XX, 3), np.uint8)  # Create a new image with double width
visRectifyV[0:YY, :, :] = imgL  # Assign the left image
visRectifyV[YY:YY * 2, :, :] = imgR  # Assign the right image

# Draw vertical lines
for x in range(0, XX, 30):
    cv2.line(visRectifyV, (x, 0), (x, YY * 2), (255, 0, 0))

visRectifyV = cv2.resize(visRectifyV, (320, 480))

mean_image = cv2.addWeighted(imgL, 0.5, imgR, 0.5, 0)
mean_image = cv2.resize(mean_image, (640, 480))

print("M: ", M)

if (visual):
    cv2.imshow('visRectifyH', visRectify)  # wizualizacja
    cv2.waitKey(0)
    cv2.imshow('visRectifyV', visRectifyV)
    cv2.waitKey(0)
    cv2.imshow('Mean', mean_image)
    cv2.waitKey(0)

