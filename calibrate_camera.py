""" Calibrate Camera

This just needs to be run once, and it will save the camera calibration matrix and distortion
coefficients in a Pickle - ./camera_pickle.p

It tests using one of the calibration files and saves the results in two test files:

1. output_images/test_undist.jpg is an undistorted version from the original matrix, testing if the 
   camera was calibrated correctly

2. After reloading the Pickle, output_images/test_reload_undist.jpg is created.  It should match #1. 
   This is testing to make sure that the Pickle was pickled correctly.  Nothing worse than a faulty 
   pickle.
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

def collect_points(images, nx, ny, show=True):
    """ Collect object and image points

    Args:
        images (list strings): A list of paths to image files
        nx (int): Number of corners in the X direction
        ny (int): Number of corners in the Y direction
        show (bool, optional): Whether or not to display the images during processing. Defaults to true.
    
    Returns:
        objpoints, imgpoints (list floats, list floats): object (3D) and image(2D) point lists
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret == True:
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners)

            if show:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
                cv2.imshow('Corners for {}'.format(fname), img)
                cv2.waitKey(500)
    
    if show:
        cv2.destroyAllWindows()
    return objpoints, imgpoints

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Collect object and image points
nx, ny = 9, 6
objpoints, imgpoints = collect_points(images, nx, ny, show=False)

# Get raw image for testing
test_img_fname = 'camera_cal/calibration3.jpg'
img = cv2.imread(test_img_fname)
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Test the calibration
dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('output_images/test_undist.jpg', dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle_file = "camera_pickle.p"
pickle.dump(dist_pickle, open(pickle_file , "wb"))

# Reload the pickle to confirm that it saved correctly
dist_unpickled = pickle.load(open(pickle_file, "rb"))
mtx_reload = dist_unpickled["mtx"]
dist_reload = dist_unpickled["dist"]

# Test the reloaded pickle
img_test = cv2.imread(test_img_fname)
dst_test = cv2.undistort(img_test, mtx_reload, dist_reload, None, mtx_reload)
cv2.imwrite('output_images/test_reload_undist.jpg', dst_test)
