import numpy as np
import cv2

# Edit this function to create your own pipeline.
def enhance_lines(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    """ Apply Color, Sobel, and Thresholds to Enhance Lines

    Apply color, Sobel, and gradient thresholds to enhance the lane lines in an RGB image 

    Args:
        img (RGB image): The image to be transformed.  *Color channels must be RGB*
        s_thresh (int pair, optional kwarg):
        The lower and upper thresholds for the S color channel after conversion to HLS.  Default=(170, 255)
        sx_thresh (int pair, optional kwarg):
        The lower and upper thresholds for the X Sobel gradient.  Default=(20, 100)
    
    Returns:
        The transformed image (binary image)
    """
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = 255*abs_sobelx/np.max(abs_sobelx)
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold S color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1.0) | (sxbinary == 1.0)] = 255
    
    return combined_binary
    
