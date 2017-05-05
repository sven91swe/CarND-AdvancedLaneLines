import cv2
import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.image as mpimg

from undistort import undistort
from transformPerspective import transformPerspective, inverseTransformPerspective
from imageFilters import threshold, sobelX, sobelY, sobelDirection, sobelMagnitude, toHLS
from findLines import findLineAndPlot

pathToTestImages = "../test_images"
outputFolder = "../output_images3"

listTestImages = os.listdir(pathToTestImages)

def pipeline(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    undistortedImage = undistort(image)


    scaledSobelX = sobelX(undistortedImage, k = 9)
    thresholdSobelX = threshold(scaledSobelX, min = 8, max=100)


    scaledSobelY = sobelY(undistortedImage, k = 9)
    thresholdSobelY = threshold(scaledSobelY, min = 8, max=100)


    sobelDir = sobelDirection(undistortedImage, k = 9)
    thresholdSobelDirection = threshold(sobelDir, min=0.7, max=1.3)


    sobelMag = sobelMagnitude(undistortedImage, k = 5)
    thresholdSobelMag = threshold(sobelMag, min=12, max=15)


    H, L, S = toHLS(undistortedImage)
    H_threshold = threshold(H, min=10, max=30)
    H_threshold_yellow = threshold(H, min=17, max=30)
    S_threshold = threshold(S, min=100, max=255)

    combined = np.zeros_like(thresholdSobelX)
    combined[(S_threshold +
              H_threshold +
              H_threshold_yellow +
              thresholdSobelMag +
              thresholdSobelDirection +
              thresholdSobelY +
              thresholdSobelX)>=6] = 1

    combinedAndTransformed = transformPerspective(combined)

    lines, radius, offMiddleBy = findLineAndPlot(None, combinedAndTransformed)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = inverseTransformPerspective(lines)
    # Combine the result with the original image
    result = cv2.addWeighted(undistortedImage, 1, newwarp, 0.3, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Radius: ' + str(int(radius)) + "m", (10, 100), font, 1, (255, 255, 255), 2)

    if offMiddleBy < 0:
        cv2.putText(result, 'Left of middle by: ' + str(-int(offMiddleBy*100)/100) + "m", (10, 300), font, 1, (255, 255, 255), 2)
    else:
        cv2.putText(result, 'Right of middle by: ' + str(int(offMiddleBy*100)/100) + "m", (10, 300), font, 1, (255, 255, 255), 2)


    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

from moviepy.editor import VideoFileClip

output = '../processed_project_video.mp4'
clip = VideoFileClip("../project_video.mp4")
input_clip = clip.fl_image(pipeline)
input_clip.write_videofile(output, audio=False)