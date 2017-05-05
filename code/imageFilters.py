"""
This file is used for various filters of images in order to dect lane lines.

Heavily based on the examples given by Udacity
"""


import numpy as np
import cv2

def threshold(singleChannelImage, min = 0, max = 255):
    sxbinary = np.zeros_like(singleChannelImage)
    sxbinary[(singleChannelImage >= min) & (singleChannelImage <= max)] = 1
    return sxbinary


def sobelX(image, k = 3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = k)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    return scaled_sobel

def sobelY(image, k = 3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = k)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    return scaled_sobel

def toHLS(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]

    return H, L, S

def sobelDirection(image, k = 3):
    absgraddir = np.arctan2(np.absolute(sobelY(image, k=k)),
                            np.absolute(sobelX(image, k=k)))
    return absgraddir

def sobelMagnitude(image, k=3):
    return np.sqrt(sobelY(image, k=k)**2 + sobelX(image, k=k)**2)