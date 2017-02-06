import numpy as np
import cv2

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
    