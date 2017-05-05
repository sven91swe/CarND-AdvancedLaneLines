import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt


old_left_fit = None
old_right_fit = None

"""
Assumes that the two lines are parallel, fit them to 2 parallel lines.
"""
def fitParallelLinesYtoX(leftX, leftY, rightX, rightY):
    left = np.transpose(np.array([leftY**2, leftY, np.ones(leftY.shape), np.zeros(leftY.shape)]))
    right = np.transpose(np.array([rightY**2, rightY, np.zeros(rightY.shape), np.ones(rightY.shape)]))

    leftHandMatrix = np.append(left, right, axis=0)

    rightHandAnswer = np.append(leftX, rightX)

    slopes = np.linalg.lstsq(leftHandMatrix, rightHandAnswer)[0]

    return slopes[[0,1,2]], slopes[[0,1,3]]


def getCurvature(leftX, leftY, rightX, rightY):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 800  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr, right_fit_cr = fitParallelLinesYtoX(leftX * xm_per_pix,
                                                     leftY * ym_per_pix,
                                                     rightX * xm_per_pix,
                                                     rightY * ym_per_pix)

    y_eval = 720
    # Calculate the new radii of curvature
    curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    # Now our radius of curvature is in meters
    return curverad

def findLineAndPlot(filename, binary_warped):

    midpoint = binary_warped.shape[1]//2
    nonzeroL = binary_warped[:, :midpoint].nonzero()
    nonzeroR = binary_warped[:, midpoint:].nonzero()

    # Extract left and right line pixel positions
    leftx = np.array(nonzeroL[1])
    lefty = np.array(nonzeroL[0])
    rightx = np.array(nonzeroR[1])+midpoint
    righty = np.array(nonzeroR[0])

    #Fit parallel polynomials to right and left line.
    left_fit, right_fit = fitParallelLinesYtoX(leftx, lefty, rightx, righty)

    global old_left_fit
    global old_right_fit

    if old_left_fit == None:
        old_left_fit = left_fit
        old_right_fit = right_fit

    """
    Checks for 'incorrect' polynomials.
    By checking that the right line is on the right side, and the left on the left side.
    Also by checking so that there is at least a certain distance between the lines.
    Does an rolling average if there is a large difference from the previous image.
    """
    thresholdToPreventChange = 30
    YPoint = 710
    left_point = left_fit[0]*YPoint**2 + left_fit[1]*YPoint + left_fit[2]
    right_point =  right_fit[0]*YPoint**2 + right_fit[1]*YPoint + right_fit[2]

    if not right_point > midpoint \
            or not left_point < midpoint:
        left_fit = old_left_fit
        right_fit = old_right_fit
    else:
        if np.abs(left_fit[2] - right_fit[2])<700:
            left_fit = old_left_fit
            right_fit = old_right_fit
        elif np.abs(old_left_fit[2]-left_fit[2])>thresholdToPreventChange \
                or np.abs(old_right_fit[2]-right_fit[2])>thresholdToPreventChange:
            left_fit = (9 * old_left_fit + left_fit)/10
            right_fit = (9 *old_right_fit + right_fit)/10
    old_left_fit = left_fit
    old_right_fit = right_fit

    left_point = left_fit[0] * YPoint ** 2 + left_fit[1] * YPoint + left_fit[2]
    right_point = right_fit[0] * YPoint ** 2 + right_fit[1] * YPoint + right_fit[2]

    """

    """
    widthOfLane = 3.7
    distanceFromLeft = (midpoint-left_point) * widthOfLane / (right_point - left_point)
    #Left is negative
    offMiddleBy = -(widthOfLane/2 - distanceFromLeft)


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    if not filename == None:
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        plt.savefig(filename)
        plt.clf()

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    radiues = getCurvature(leftx, lefty, rightx, righty)


    return color_warp, radiues, offMiddleBy

