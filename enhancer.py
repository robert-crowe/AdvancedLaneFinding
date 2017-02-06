import numpy as np
import cv2
import pickle
from perspective import Perspective
from scipy.signal import find_peaks_cwt
import matplotlib.pyplot as plt
from slider import moving_average

# Edit this function to create your own pipeline.
def enhance_lines(img, HLS_thresh=(170, 255)):
    """ Apply Color, Sobel, and Thresholds to Enhance Lines

    Apply color, Sobel, and gradient thresholds to enhance the lane lines in an RGB image 

    Args:
        img (RGB image): The image to be transformed.  *Color channels must be BGR*
        HLS_thresh (int pair, optional kwarg):
        The lower and upper thresholds for the S color channel after conversion to HLS.  Default=(170, 255)
    
    Returns:
        The transformed image (binary image)
    """
    # Convert to HSV color space and separate the channels
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # .astype(np.float)
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]

    # Mask the yellows
    # low  = np.array([90, 85, 0])
    # high = np.array([100, 255, 255])
    low  = np.array([90, 85, 0])
    high = np.array([101, 255, 255])
    yellows = cv2.inRange(hsv, low, high)
    cv2.imshow('Yellows', yellows)

    # Mask the whites
    low  = np.array([0, 0, 200])
    high = np.array([255, 25, 255])
    whites = cv2.inRange(hsv, low, high)
    cv2.imshow('Whites', whites)

    # Convert to HLS and threshold S
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hsv[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= HLS_thresh[0]) & (s_channel < HLS_thresh[1])] = 1
    s_binary = np.asarray(s_binary * 255, dtype=np.uint8)
    cv2.imshow('s_binary', s_binary)

    # Combine the two binary thresholds
    combined_binary = cv2.bitwise_or(yellows, cv2.bitwise_or(whites, s_binary))
    
    return combined_binary

def get_enhanced_birdseye(image):
    """ Transform to a birdseye view and enhance to find the lines

    Args:
        image (image): The original image from the camera
    
    Returns:
        birdseye (image): The enhanced birdseye view
    """
    cv2.imshow('Original', image)

    # Enhance the lines
    # birdseye = enhance_lines(birdseye, HLS_thresh=[175, 250])
    image = enhance_lines(image, HLS_thresh=[200, 255])
    cv2.imshow('Enhanced', image)
    return image

def transform2birdseye(image):
    # Transform it to birdseye
    P = Perspective()
    birdseye = P.to_birdseye(image)
    cv2.imshow('Birdseye view', birdseye)
    print('birdseye shape: {}'.format(birdseye.shape))
    return birdseye

def correct_image(img):
    """ Correct for camera distortion.  You should only do this once per camera image. 

    Args:
        img (image): The image to be corrected
    
    Returns:
        img (image): The corrected image
    """
    # Reload the pickle
    dist_unpickled = pickle.load(open("camera_pickle.p", "rb"))
    mtx = dist_unpickled["mtx"]
    dist = dist_unpickled["dist"]

    # Correct the camera
    return cv2.undistort(img, mtx, dist, None, mtx)

def find_lines(img):
    """ Starting from the moving average from the enhanced birdseye, try to find two lane lines. 

    Args:
        img (image): The enhanced birdseye
    
    Returns:
        peaks (list int): The complete list of peaks found in the final pass
        nice_peaks (list int): The X coordinates of the best two peaks found
        both_lines (bool): Did we find two good lines?
    """
    # Get the moving average
    avg_width = 25
    Mavg = moving_average(img, width=avg_width)

    # Find the nicest peaks in the moving average
    typical_line_maxWpx = 50
    typical_lane_maxWpx = 750
    peak_thresh = 0.5
    step = 0.05
    num_peaks = 0
    found_twins = True
    nice_peaks = []
    while found_twins and len(nice_peaks) is 0: 
        peaks = find_peaks_cwt(Mavg > peak_thresh, [typical_line_maxWpx], max_distances=[typical_lane_maxWpx])
        num_peaks = len(peaks)
        peak_thresh -= step
        if peak_thresh < 0 and num_peaks < 2:
            found_twins = False
        elif num_peaks > 1:
            for p1 in peaks:  # did we find a pair about the right distance apart to be a lane?
                for p2 in peaks:
                    dist = np.absolute(p2 - p1)
                    if dist > 600 and dist < 800:
                        nice_peaks = [p1, p2]
                        break
                if len(nice_peaks) > 0:
                    break

    both_lines = found_twins and len(nice_peaks) is 2
    
    plt.plot(Mavg) # DEBUG
    plt.show()

    return peaks, nice_peaks, both_lines