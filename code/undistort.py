"""
This file is used for undistorting a image.

Heavily based on the examples given by Udacity
"""


import cv2
import os
import numpy as np

relativePath = "../camera_cal"
nx = 9
ny = 6

objectPoints = []
imagePoints = []
listCalibrationImages = os.listdir(relativePath)

objp = np.zeros((nx * ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

for imageName in listCalibrationImages:
    image = cv2.imread(relativePath+'/'+imageName)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret:
        imagePoints.append(corners)
        objectPoints.append(objp)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, gray.shape[::-1], None, None)


def undistort(imageToBeUndistorted):
    return cv2.undistort(imageToBeUndistorted, mtx, dist, None, mtx)
