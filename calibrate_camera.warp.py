""" Calibrate Camera - Warp Test

This is a test to make sure that our camera calibration works correctly with 
perspective transformation.  It's easier to check this with a chessboard
to make sure before we apply it to real images.
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

# Define a function that takes an image, number of x and y points, 
# camera matrix and distortion coefficients
def corners_unwarp(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M

# Get raw image for testing
test_img_fname = 'camera_cal/calibration9.jpg'
img = cv2.imread(test_img_fname)
img_size = (img.shape[0:2])

# Load the pickle to get the matrix and coefficients
pickle_file = "camera_pickle.p"
dist_unpickled = pickle.load(open(pickle_file, "rb"))

mtx = dist_unpickled["mtx"]
dist = dist_unpickled["dist"]

# Warp the image and display
nx, ny = 9, 6
dst, M = corners_unwarp(img, nx, ny, mtx, dist)

cv2.imshow('Warped {}'.format(test_img_fname), dst)
cv2.waitKey(10000)