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
    cv2.waitKey(1000)

    ret, otsu_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imshow('Otsu Gray', otsu_gray) # DEBUG
    cv2.waitKey(1000)

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]

    cv2.imshow('S', S) # DEBUG
    cv2.waitKey(1000)

    ret, otsu_S = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imshow('Otsu S', otsu_S) # DEBUG
    cv2.waitKey(1000)

    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    cv2.imshow('R', R) # DEBUG
    cv2.waitKey(1000)

    ret, otsu_R = cv2.threshold(R, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imshow('Otsu R', otsu_R) # DEBUG
    cv2.waitKey(1000)

    RorS = cv2.bitwise_or(R, S)
    ret, otsu_RorS = cv2.threshold(RorS, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('Otsu R or S', otsu_RorS) # DEBUG
    cv2.waitKey(1000)

    RandS = cv2.bitwise_and(R, S)
    ret, otsu_RandS = cv2.threshold(RandS, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('Otsu R and S', otsu_RandS) # DEBUG
    cv2.waitKey(1000)

    # blur = cv2.GaussianBlur(gray, (7,7), 0)
    # ret, otsu_blur_gray = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # cv2.imshow('Otsu Blur Gray', otsu_blur_gray) # DEBUG
    # cv2.waitKey(1000)
    
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient is 'x':
        sobel = cv2.Sobel(otsu_gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(otsu_gray, cv2.CV_64F, 0, 1)
    
    # sobelx = cv2.Sobel(otsu_blur_gray, cv2.CV_64F, 1, 0)
    # cv2.imshow('Sobel Otsu X', sobelx) # DEBUG
    # cv2.waitKey(1000)

    # sobely = cv2.Sobel(otsu_blur_gray, cv2.CV_64F, 0, 1)
    # cv2.imshow('Sobel Otsu Y', sobely) # DEBUG
    # cv2.waitKey(1000)

    # cv2.imshow('Sobel Otsu', sobel) # DEBUG
    # cv2.waitKey(1000)

    # # Take the absolute value of the derivative or gradient
    # abs_sobel = np.absolute(sobel)

    # cv2.imshow('Absolute Sobel', abs_sobel) # DEBUG
    # cv2.waitKey(5000)

    # # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # cv2.imshow('Scaled Absolute Sobel', scaled_sobel) # DEBUG
    # cv2.waitKey(5000)

    # Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    # sbinary = np.zeros_like(scaled_sobel)
    # sbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # cv2.imshow('Masked Scaled Absolute Sobel', sbinary) # DEBUG
    # cv2.waitKey(5000)

    # Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(scaled_sobel, (3,3), 0)
    # ret, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # cv2.imshow('Otsu with Gauss', otsu) # DEBUG
    # cv2.waitKey(5000)

    # Return this mask as the output image
    return sobel
