import numpy as np
from scipy.signal import find_peaks_cwt
import math
import cv2

def histo(img):
    return np.sum(img, axis=0)

def half_img(img, window_height):
    half_height = np.uint16(window_height/2)
    end = img.shape[0]
    start = end - half_height
    # return img[start:end,:,:]
    return img[start:end,:]

def make_box(img, H, W, Ybottom, Xcenter):
    Ybottom = np.uint16(Ybottom)
    Xstart = np.uint16(Xcenter - W // 2)
    Xend = np.uint16(Xstart + W)
    Ytop = np.uint16(Ybottom - H)
    return img[Ytop:Ybottom, Xstart:Xend]

def moving_average(img, width=32):
    # adapted from http://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    cols = np.mean(img[:,:], axis=0)
    Mavg = np.cumsum(cols, dtype=float)
    Mavg[width:] = Mavg[width:] - Mavg[:-width]
    col_max = np.max(cols)
    if (col_max > 0):
        Mavg = (Mavg[width - 1:] / width) / np.max(cols)
        return Mavg / np.max(Mavg) # scale to 0.0 - 1.0
    else:
        Mavg = cols # zeros
        return Mavg

def find_box_peak(img):
    avg_line_width_px = 25
    range_line_widths = [100]
    max_dist = [800]
    peak_thresh = 0.5
    # Get the moving average
    Mavg = moving_average(img, width=avg_line_width_px)

    # Find the nicest peaks in the moving average
    peaks = find_peaks_cwt(Mavg > peak_thresh, range_line_widths, max_distances=max_dist)
    if len(peaks) == 0:
        return None
    else:
        mean_peaks = np.mean(peaks)
        if math.isnan(mean_peaks):
            return None 
        else:
            return mean_peaks

def walk_lines(last_good, enhanced):
    left_start = float(last_good[0])
    right_start = float(last_good[1])
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

    # Walk up the lines
    while curY >= walk_Y:
        left_box = make_box(enhanced, walk_Y, box_width, curY, leftBoxCenter)
        right_box = make_box(enhanced, walk_Y, box_width, curY, rightBoxCenter)
        # cv2.imshow('left_box', left_box)
        # cv2.imshow('right_box', right_box)

        found_left = find_box_peak(left_box) # peak relative to box
        if found_left is not None: # did we find a peak?
            nextLeftX = float(max(found_left + leftBoxCenter - box_half, minX)) # don't go past the left border
            leftDeltaSteps.append(curLeftX - nextLeftX) # keep track of the deltas for averaging steps
            curLeftX = nextLeftX
        elif len(leftDeltaSteps) > 0:
            # take a step in the average direction - we can lose dashed lines if we just go straight up
            curLeftX = float(max(curLeftX - np.mean(leftDeltaSteps), 0))
        leftBoxCenter = min(max(curLeftX, minX), maxX)

        found_right = find_box_peak(right_box)
        if found_right is not None:
            nextRightX = float(min(found_right + rightBoxCenter - box_half, maxX))
            rightDeltaSteps.append(curRightX - nextRightX)
            curRightX = nextRightX
        elif len(rightDeltaSteps) > 0:
            curRightX = float(min(curRightX - np.mean(rightDeltaSteps), maxX))
        rightBoxCenter = max(min(curRightX, maxX), minX)
        
        left_Xpts.append(curLeftX)
        right_Xpts.append(curRightX)
        Ypts.append(float(curY + half_walk)) # Y in middle, not top edge of box

        curY = curY - walk_Y # take the next step
        # cv2.waitKey(0)  # DEBUG

    return left_Xpts, right_Xpts, Ypts