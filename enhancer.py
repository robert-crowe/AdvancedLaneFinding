import numpy as np
import cv2

# Edit this function to create your own pipeline.
def enhance_lines(img, HLS_thresh=(170, 255), Gray_thresh=(20, 100)):
    """ Apply Color, Sobel, and Thresholds to Enhance Lines

    Apply color, Sobel, and gradient thresholds to enhance the lane lines in an RGB image 

    Args:
        img (RGB image): The image to be transformed.  *Color channels must be BGR*
        HLS_thresh (int pair, optional kwarg):
        The lower and upper thresholds for the S color channel after conversion to HLS.  Default=(170, 255)
        Gray_thresh (int pair, optional kwarg):
        The lower and upper thresholds for the X Sobel gradient.  Default=(20, 100)
    
    Returns:
        The transformed image (binary image)
    """
    # Convert to HSV color space and separate the channels
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:,:,0]
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cv2.imshow('Grayscale', gray)

    # Sobel x grayscale
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = 255*abs_sobelx/np.max(abs_sobelx)
    
    # Threshold x gradient grayscale
    gray_binary = np.zeros_like(scaled_sobel, dtype=np.float)
    gray_binary[(scaled_sobel >= Gray_thresh[0]) & (scaled_sobel <= Gray_thresh[1])] = 1
    # cv2.imshow('Sobel', gray_binary)
    
    # Threshold S color channel
    S_binary = cv2.inRange(s_channel, HLS_thresh[0], HLS_thresh[1]).astype(np.float)
    # cv2.imshow('S Channel', S_binary)

    # Combine the two binary thresholds
    combined_binary = cv2.bitwise_or(gray_binary, S_binary)
    
    return np.asarray(combined_binary * 255, dtype=np.uint8)
    