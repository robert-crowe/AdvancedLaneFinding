import numpy as np
from scipy.signal import find_peaks_cwt
import math
import cv2

def histo(img):
    """ Generate a histogram from a 2d black and white image. 

    Args:
        img (image): The input image

    Returns:
        histo (1d array): The histogram of the image
    """
    return np.sum(img, axis=0)

def half_img(img, window_height):
    """ Get the bottom half of an image

    Args:
        img (image): The input image
        window_height (int): The starting height of the image
    
    Returns:
        img (image): The bottom half of the image
    """
    half_height = np.uint16(window_height/2)
    end = img.shape[0]
    start = end - half_height
    # return img[start:end,:,:]
    return img[start:end,:]

def make_box(img, H, W, Ybottom, Xcenter):
    """ Make a sliding window box

    Args:
        img (image): The entire binary thresholded image
        H (int): The desired height of the box
        W (int): The desired width of the box
        Ybottom (int): The Y coordinate of the bottom of the box
        Xcenter (int): The X coordinate of the center of the box
    
    Returns:
        img (image): The box
    """
    Ybottom = np.uint16(Ybottom)
    Xstart = np.uint16(Xcenter - W // 2)
    Xend = np.uint16(Xstart + W)
    Ytop = np.uint16(Ybottom - H)
    return img[Ytop:Ybottom, Xstart:Xend]

def moving_average(img, width=32):
    """ Calculate a moving average of a 1D array and scale

    Args:
        img (image): The 1d array
        width (int, optional kwarg): The width of the interval
    
    Returns:
        Mavg (1d array): The moving average of the input array, scaled to 0.0 to 1.0
    """
    # adapted from http://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    cols = np.mean(img[:,:], axis=0)
    Mavg = np.cumsum(cols, dtype=float)
    Mavg[width:] = Mavg[width:] - Mavg[:-width]
    if len(cols) > 0:
        col_max = np.max(cols)
    else:
        col_max = 0
    if (col_max > 0):
        Mavg = (Mavg[width - 1:] / width) / np.max(cols)
        return Mavg / np.max(Mavg) # scale to 0.0 - 1.0
    else:
        Mavg = cols # zeros
        return Mavg

def find_box_peak(img):
    """ Find the strongest signal in a sliding window, which should correspond to a line if one is present

    Args:
        img (image): The sliding window
    
    Returns:
        mean_peaks (int): The X location of the strongest signal
    """
    avg_line_width_px = 25
    range_line_widths = [100]
    max_dist = [800]
    peak_thresh = 0.5
    # Get the moving average
    Mavg = moving_average(img, width=avg_line_width_px)

    # Find the nicest peaks in the moving average
    if len(Mavg > peak_thresh) > 0:
        peaks = find_peaks_cwt(Mavg > peak_thresh, range_line_widths, max_distances=max_dist)
    else:
        peaks = []
    if len(peaks) == 0:
        return None
    else:
        mean_peaks = np.mean(peaks)
        if math.isnan(mean_peaks):
            return None 
        else:
            return mean_peaks

def walk_lines(enhanced, gEnv):
    """ Walk up a suspected lane line from bottom to top, using a sliding window to locate the line

    'Because you're mine, I walk the line'

    Args:
        enhanced (image): A binary thresholded image of a view of the lane from above
        gEnv (dict): The global configuration and state
    
    Returns:
        left_Xpts (array): The X coordinates for the left line
        right_Xpts (array): The X coordinates for the right line
        Ypts (array): The Y coordinates for both lines (they're the same for both)
        data_good (bool): Did we find decent enough data?
    """
    left_start = float(gEnv['prev_line_ends'][0])
    right_start = float(gEnv['prev_line_ends'][1])
    left_coef = gEnv['left_prev_coef']
    right_coef = gEnv['right_prev_coef']
    walk_Y = 36
    half_walk = walk_Y // 2
    box_width = 200
    box_half = box_width // 2
    curY = enhanced.shape[0]
    minX = box_half # from the center of the box, don't go past the left edge of the image
    maxX = enhanced.shape[1] - minX # same, right edge
    curLeftX = float(max(left_start, minX)) # don't start out already past the left edge
    curRightX = float(min(right_start, maxX)) # same, right edge
    leftBoxCenter = curLeftX
    rightBoxCenter = curRightX
    left_Xpts = []
    right_Xpts = []
    Ypts = []
    leftDeltaSteps = []
    rightDeltaSteps = []
    steps = 0
    left_peaks_found = 0
    right_peaks_found = 0

    # Walk up the lines
    while curY >= walk_Y:
        steps += 1
        left_box = make_box(enhanced, walk_Y, box_width, curY, leftBoxCenter)
        right_box = make_box(enhanced, walk_Y, box_width, curY, rightBoxCenter)
        # cv2.imshow('left_box', left_box)
        # cv2.imshow('right_box', right_box)

        found_left = find_box_peak(left_box) # peak relative to box
        if found_left is not None: # did we find a peak?
            left_peaks_found += 1
            nextLeftX = float(max(found_left + leftBoxCenter - box_half, minX)) # don't go past the left border
            leftDeltaSteps.append(curLeftX - nextLeftX) # keep track of the deltas for averaging steps
            curLeftX = nextLeftX
        elif gEnv['prev_data_good'] and left_coef is not None: # if we don't see a line, use the previous coefficients 
            curLeftX = (left_coef[0] * curY**2) + (left_coef[1] * curY) + left_coef[2]
        elif len(leftDeltaSteps) > 0: # we don't have previous coefficients, so use the recent trend
            # take a step in the average direction - we can lose dashed lines if we just go straight up
            curLeftX = float(max(curLeftX - np.mean(leftDeltaSteps), 0))
        leftBoxCenter = min(max(curLeftX, minX), maxX)

        found_right = find_box_peak(right_box)
        if found_right is not None:
            right_peaks_found += 1
            nextRightX = float(min(found_right + rightBoxCenter - box_half, maxX))
            rightDeltaSteps.append(curRightX - nextRightX)
            curRightX = nextRightX
        elif gEnv['prev_data_good'] and right_coef is not None:
            curRightX = (right_coef[0] * curY**2) + (right_coef[1] * curY) + right_coef[2]
        elif len(rightDeltaSteps) > 0:
            curRightX = float(min(curRightX - np.mean(rightDeltaSteps), maxX))
        rightBoxCenter = max(min(curRightX, maxX), minX)
        
        left_Xpts.append(curLeftX)
        right_Xpts.append(curRightX)
        Ypts.append(float(curY + half_walk)) # Y in middle, not top edge of box

        curY = curY - walk_Y # take the next step
        # cv2.waitKey(0)  # DEBUG
    
    left_ratio_good = left_peaks_found / steps
    right_ratio_good = right_peaks_found / steps
    data_good = False
    min_good = 0.1
    if left_ratio_good > min_good and right_ratio_good > min_good:
        data_good = True

    return left_Xpts, right_Xpts, Ypts, data_good