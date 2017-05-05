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
outputFolder = "../output_images"

listTestImages = os.listdir(pathToTestImages)

image = cv2.imread("../camera_cal/calibration1.jpg")
undistortedImage = undistort(image)
cv2.imwrite(outputFolder + "/" + "undist-chessboard.jpg", undistortedImage)



for imageName in listTestImages:
    #image = cv2.imread(pathToTestImages + "/" + imageName)

    image = mpimg.imread(pathToTestImages + "/" + imageName)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    pathAndNameForOutput = outputFolder + "/"
    nameWithoutFormat = imageName.split(".")[0]

    cv2.imwrite(pathAndNameForOutput + "original-" + imageName, image)


    undistortedImage = undistort(image)
    cv2.imwrite(pathAndNameForOutput + "undist-" + imageName, undistortedImage)

    transformed = transformPerspective(undistortedImage)
    cv2.imwrite(pathAndNameForOutput + "transformed-" + imageName, transformed)



    scaledSobelX = sobelX(undistortedImage, k = 9)
    cv2.imwrite(pathAndNameForOutput + "sobelX-" + imageName, scaledSobelX)

    thresholdSobelX = threshold(scaledSobelX, min = 8, max=100)
    #cv2.imwrite(pathAndNameForOutput + "sobelX - threshold - " + imageName, thresholdSobelX)
    plt.imsave(pathAndNameForOutput + "sobelX-threshold-" + nameWithoutFormat + ".png", thresholdSobelX,
               cmap = 'gray', format='png')


    scaledSobelY = sobelY(undistortedImage, k = 9)
    cv2.imwrite(pathAndNameForOutput + "sobelY-" + imageName, scaledSobelY)

    thresholdSobelY = threshold(scaledSobelY, min = 8, max=100)
    plt.imsave(pathAndNameForOutput + "sobelY-threshold-" + nameWithoutFormat + ".png", thresholdSobelY,
               cmap = 'gray', format='png')

    sobelDir = sobelDirection(undistortedImage, k = 9)
    plt.imsave(pathAndNameForOutput + "sobelDirection-" + nameWithoutFormat + ".png", sobelDir,
               cmap='gray', format='png')

    thresholdSobelDirection = threshold(sobelDir, min=0.7, max=1.3)
    plt.imsave(pathAndNameForOutput + "sobelDirection-threshold-" + nameWithoutFormat + ".png", thresholdSobelDirection,
               cmap='gray', format='png')

    sobelMag = sobelMagnitude(undistortedImage, k = 5)
    plt.imsave(pathAndNameForOutput + "sobelMagnitude-" + nameWithoutFormat + ".png", sobelMag,
               cmap='gray', format='png')

    thresholdSobelMag = threshold(sobelMag, min=12, max=15)
    plt.imsave(pathAndNameForOutput + "sobelMagnitude-threshold-" + nameWithoutFormat + ".png", thresholdSobelMag,
               cmap='gray', format='png')

    H, L, S = toHLS(undistortedImage)
    cv2.imwrite(pathAndNameForOutput + "hue-" + imageName, H)
    cv2.imwrite(pathAndNameForOutput + "lightness-" + imageName, L)
    cv2.imwrite(pathAndNameForOutput + "saturation-" + imageName, S)

    H_threshold = threshold(H, min=10, max=30)
    plt.imsave(pathAndNameForOutput + "hue-threshold-" + nameWithoutFormat + ".png",
               H_threshold,
               cmap='gray', format='png')

    H_threshold_yellow = threshold(H, min=17, max=30)
    plt.imsave(pathAndNameForOutput + "hue_yellow-threshold-" + nameWithoutFormat + ".png",
               H_threshold_yellow,
               cmap='gray', format='png')


    S_threshold = threshold(S, min=100, max=255)
    plt.imsave(pathAndNameForOutput + "saturation-threshold-" + nameWithoutFormat + ".png",
               S_threshold,
               cmap='gray', format='png')

    L_threshold = threshold(L, min=100, max=255)
    plt.imsave(pathAndNameForOutput + "lightness-threshold-" + nameWithoutFormat + ".png",
               L_threshold,
               cmap='gray', format='png')

    combined = np.zeros_like(thresholdSobelX)
    combined[(S_threshold +
              H_threshold +
              H_threshold_yellow +
              thresholdSobelMag +
              thresholdSobelDirection +
              thresholdSobelY +
              thresholdSobelX)>=6] = 1

    plt.imsave(pathAndNameForOutput + "combined-" + nameWithoutFormat + ".png",
               combined,
               cmap='gray', format='png')

    combinedAndTransformed = transformPerspective(combined)
    plt.imsave(pathAndNameForOutput + "combinedAndTransformed-" + nameWithoutFormat + ".png",
               combinedAndTransformed,
               cmap='gray', format='png')

    print(imageName)
    import findLines

    findLines.old_left_fit = None
    findLines.old_right_fit = None

    lines, radius, offMiddleBy = findLineAndPlot(pathAndNameForOutput + "foundLine-" + nameWithoutFormat + ".png", combinedAndTransformed)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = inverseTransformPerspective(lines)
    # Combine the result with the original image
    result = cv2.addWeighted(undistortedImage, 1, newwarp, 0.3, 0)
    cv2.imwrite(pathAndNameForOutput + "resultWithLines-" + imageName, result)


    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Radius: ' + str(radius) + "m", (10, 500), font, 1, (255, 255, 255), 2)

    if offMiddleBy < 0:
        cv2.putText(result, 'Left of middle by: ' + str(offMiddleBy) + "m", (10, 100), font, 1, (255, 255, 255), 2)
    else:
        cv2.putText(result, 'Right of middle by: ' + str(offMiddleBy) + "m", (10, 300), font, 1, (255, 255, 255), 2)


    convertedRGB = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    plt.imsave(pathAndNameForOutput + "convertedRGB-" + nameWithoutFormat + ".png",
               convertedRGB, format='png')

