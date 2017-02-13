import numpy as np
from scipy.signal import find_peaks_cwt
import matplotlib.pyplot as plt
from slider import moving_average
from slider import half_img, histo
from enhancer import get_enhanced
import cv2

def find_line_starts(image, gEnv):
    """ Find the beginnings of the lane lines closest to the car

    This does a walk down the image, starting with the full image and taking the bottom half at each
    iteration until it reaches a minimum size.  By starting with the whole image we account for gaps
    in dashed lines, but as we walk down we account for line curvature towards the top of the image. 
    If we can find the lines in the last slice closest to the car, they are likely to be the most
    accurate, so those pixels are included in all iterations. 

    Args:
        img (img): The RGB birdseye view after correction and contrast adjustment
        gEnv: Global config, including the previous starts of the lines, or None if need to find
    
    Returns:
        last_good (list of 2): The means of the left and right lines found
        enhanced (image): The color masked, Sobel enhanced image
    """
    window_height = image.shape[0]
    min_height = 179
    good_list = []
    last_good = gEnv['prev_line_ends']
    # get the enhanced image
    enhanced = get_enhanced(image, gEnv)
    if not gEnv['prev_data_good']:
        window = enhanced
        while window_height > min_height:
            # What does the histogram look like?  (DEBUG)
            # histogram = histo(window)
            # plt.plot(histogram)
            # plt.show()

            # Try to find two good lines
            peaks, nice_peaks = find_lines(window)
            # print('Peaks: {}, Nice_Peaks: {}'.format(peaks, nice_peaks))
            if nice_peaks is not None:
                good_list.append(nice_peaks)

            # Grab the lower half
            window = half_img(window, window_height)
            window_height = window.shape[0]
            # cv2.imshow('Window', window)

        if len(good_list) > 0:
            good_list = np.reshape(good_list, (len(good_list), 2))
            last_good = [int(np.mean(good_list[:, 0])), int(np.mean(good_list[:, 1]))]
        
    return enhanced, last_good

def find_lines(img):
    """ Starting from the moving average from the enhanced birdseye, try to find two lane lines. 

    This sets a progressively lower threshold for peaks in the moving average at each iteration,
    until it reaches a minimum.  We want to detect the stronger peaks first, but not miss weaker
    peaks altogether.

    Peaks are not always found in pairs, because of dashed lines, shadows, bright spots, etc.
    So we sometimes need to extrapolate the other line when we only have a single peak.

    Args:
        img (image): The enhanced birdseye
    
    Returns:
        peaks (list int): The complete list of peaks found in the final pass
        best_peaks (list int): The X coordinates of the best two peaks found
    """
    # some parameters
    avg_line_width_px = 25
    typical_lane_width_px = 885  # in birdseye, just past the hood
    center_left_line_area = 197
    center_right_line_area = 1082

    # Get the moving average
    Mavg = moving_average(img, width=avg_line_width_px)

    # Find the nicest peaks in the moving average
    range_line_widths = [100]
    # range_line_widths = [90, 100, 110]
    max_dist = [800]
    # max_dist = [700, 800, 900]
    peak_thresh = 0.5
    step = 0.05
    num_peaks = []
    nice_peaks = []
    single_lefts = []
    single_rights = []
    min_thresh = 0.2
    while peak_thresh > min_thresh: 
        peaks = find_peaks_cwt(Mavg > peak_thresh, range_line_widths, max_distances=max_dist)
        left_peaks = list(filter((lambda x: x < 640), peaks))
        right_peaks = list(filter((lambda x: x >= 640), peaks))
        num_peaks = len(peaks)
        peak_thresh = peak_thresh - step
        # prefer to find both lines in one slice
        if num_peaks > 0:
            if len(left_peaks) > 0 and len(right_peaks) > 0:
                nice_peak = find_best_pair(left_peaks, right_peaks, center_left_line_area, center_right_line_area)
                if len(nice_peak) > 0:
                    nice_peaks.append(nice_peak)
            elif len(left_peaks) > 0: # left line
                for peak in left_peaks:
                    single_lefts.append(peak)
            elif len(right_peaks) > 0: # right line
                for peak in right_peaks:
                    single_rights.append(peak)
    
    best_peaks = None
    if len(nice_peaks) > 0:
        # find the pair closest to lane width apart
        best_peaks = best_line_width(nice_peaks, typical_lane_width_px)

    else: # need to work with single peaks
        if len(single_lefts) > 0 and len(single_rights) > 0:  # we have some of each
            best_peaks = find_best_pair(single_lefts, single_rights, center_left_line_area, center_right_line_area)
        elif len(single_lefts) > 0: # only have left lines
            # find the one closest to the center of the left area and extrapolate right
            left = find_closest_peak(single_lefts, center_left_line_area)
            best_peaks = [left, left + typical_lane_width_px]
        elif len(single_rights) > 0: # only have right lines
            # find the one closest to the center of the right area and extrapolate left
            right = find_closest_peak(single_rights, center_right_line_area)
            best_peaks = [right - typical_lane_width_px, right]
    
    # plt.plot(Mavg) # DEBUG
    # plt.show()
    return peaks, best_peaks

def jagged_flat(arr):
    """ Flattening a jagged array
    """
    return (lambda l: [item for sublist in l for item in sublist])(arr)

def find_best_pair(left_peaks, right_peaks, center_left_line_area, center_right_line_area):
    """ Find a matched pair of peaks in about the right location, about the right distance apart

    Args:
        left_peaks (array int): The peaks found on the left
        right_peaks (array int): The peaks found on the right 
        center_left_line_area (int): The center of the area where the left line will be 
        center_right_line_area (int): The center of the area where the right line will be
    
    Returns:
        best_peaks (arry of 2 ints): The best peaks found
    """
    pairs_list, left_lines, right_lines = [], [], []
    left_peaks = np.asarray(left_peaks).flatten()
    right_peaks = np.asarray(right_peaks).flatten()

    for left in left_peaks:  # find the left line closest to the center of the left area
        delta = abs(center_left_line_area - left)
        left_lines.append((delta, left))

    ordered_lefts = sorted(left_lines, key=lambda line: line[0])
    closest_left = ordered_lefts[0]

    for right in right_peaks:  # find the right line closest to the center of the right area
        delta = abs(center_right_line_area - right)
        right_lines.append((delta, right))

    ordered_rights = sorted(right_lines, key=lambda line: line[0])
    closest_right = ordered_rights[0]

    return [closest_left[1], closest_right[1]]

def find_closest_peak(peaks, center_px):
    """ Find the peak closest to the center of the area

    Args:
        peaks (array int): The peaks to check
        center_px (int): The center pixel of the area
    
    Returns:
        peak (int): The closest peak
    """
    peaks_list = []
    # flat_peaks = (lambda l: [item for sublist in l for item in sublist])(peaks)
    flat_peaks = peaks
    unique_peaks = np.unique(flat_peaks)
    for peak in unique_peaks:
        dist = abs(center_px - peak)
        peaks_list.append((dist, peak))
    ordered_peaks = sorted(peaks_list, key=lambda peak: peak[0])
    return ordered_peaks[0][1]

def best_line_width(nicest_peaks, typical_lane_width_px):
    """ Test pairs of peaks to try to find the ones closest to the width of a lane apart

    Args:
        nicest_peaks (array pairs ints): Pairs of peaks to test
        typical_lane_width_px (int): The typical width of a lane in pixels
    
    Returns:
        closest (pair ints): The closest pair of peaks
    """
    pairs_list = []
    for peak in nicest_peaks:  # did we find a pair about the right distance apart to be a lane?
        left, right = peak[0], peak[1]
        dist = right - left
        delta = abs(typical_lane_width_px - dist)
        pairs_list.append((delta, (left, right)))
    ordered_pairs = sorted(pairs_list, key=lambda pair: pair[0])
    closest_typical = ordered_pairs[0]
    return closest_typical[1]