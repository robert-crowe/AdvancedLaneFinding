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
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    cv2.imshow('Gray', gray) # DEBUG
    cv2.waitKey(5000)

    ret, otsu_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imshow('Gray', otsu_gray) # DEBUG
    cv2.waitKey(5000)
    
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient is 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
    cv2.imshow('Sobel', sobel) # DEBUG
    cv2.waitKey(5000)

    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    cv2.imshow('Absolute Sobel', abs_sobel) # DEBUG
    cv2.waitKey(5000)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    cv2.imshow('Scaled Absolute Sobel', scaled_sobel) # DEBUG
    cv2.waitKey(5000)

    # Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    # sbinary = np.zeros_like(scaled_sobel)
    # sbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # cv2.imshow('Masked Scaled Absolute Sobel', sbinary) # DEBUG
    # cv2.waitKey(5000)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(scaled_sobel, (3,3), 0)
    ret, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cv2.imshow('Otsu with Gauss', otsu) # DEBUG
    cv2.waitKey(5000)

    # Return this mask as the output image
    return otsu
