import numpy as np
from scipy.signal import find_peaks_cwt
import math

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
