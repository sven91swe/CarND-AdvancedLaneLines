"""
This file is used for transforming the perspective to and from a 'birds-eye' view.

Heavily based on the examples given by Udacity
"""


import cv2
import numpy as np

"""
offset650 = 250
offset475 = 544
src=[[offset650, 650],
     [offset475, 475],
     [1280-offset475, 475],
     [1280-offset650, 650]]

src = np.float32(src)

"""

"""
offset650 = 250
offset450 = 590
src=[[offset650, 650],
     [offset450, 450],
     [1280-offset450, 450],
     [1280-offset650, 650]]

src = np.float32(src)
"""
offset690 = 213
offset450 = 590
src=[[offset690, 690],
     [offset450, 450],
     [1280-offset450, 450],
     [1280-offset690, 690]]

src = np.float32(src)


offsetX = 200
dst=[[offsetX, 720],
     [offsetX, 0],
     [1280-offsetX, 0],
     [1280-offsetX, 720]]

dst = np.float32(dst)


M = cv2.getPerspectiveTransform(src, dst)

Minv = cv2.getPerspectiveTransform(dst, src)


def transformPerspective(image):
    img_size = (image.shape[1], image.shape[0])
    return cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)


def inverseTransformPerspective(image):
    img_size = (image.shape[1], image.shape[0])
    return cv2.warpPerspective(image, Minv, img_size, flags=cv2.INTER_LINEAR)
