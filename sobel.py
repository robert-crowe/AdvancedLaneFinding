import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    """ Apply Sobel Transform

    Apply a Sobel transform in either X or Y direction, and return the absolute value 
    of the gradient. 

    Args:
        img (RGB image): The image to be transformed.  *Color channels must be RGB*
        orient ('x' or 'y', optional kwarg): The direction of the transform.  Default = 'x'
        thresh_min (int, optional kwarg): The minimum threshold.  Default = 0
        thresh_max (int, optional kwarg): The maximum threshold.  Default = 255
    
    Returns:
        The transformed image (binary image)
    """
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient is 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    min = np.min(scaled_sobel)
    max = np.max(scaled_sobel)
    print('Min: {}, Max: {}'.format(min, max))
    
    mask = np.zeros_like(scaled_sobel)
    mask[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    masked_image = cv2.bitwise_and(gray, mask)
    
    # 6) Return this mask as your binary_output image
    # binary_output = np.copy(img) # Remove this line
    return masked_image
