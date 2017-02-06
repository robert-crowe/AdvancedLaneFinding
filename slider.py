import numpy as np

def histo(img):
    return np.sum(img, axis=0)

def half_img(img, window_height):
    half_height = np.uint16(window_height/2)
    end = img.shape[0]
    start = end - half_height
    return img[start:end,:,:]

def moving_average(img, width=32):
    # adapted from http://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    cols = np.mean(img[:,:], axis=0)
    Mavg = np.cumsum(cols, dtype=float)
    Mavg[width:] = Mavg[width:] - Mavg[:-width]
    Mavg = (Mavg[width - 1:] / width) / np.max(cols)
    return Mavg / np.max(Mavg) # scale to 0.0 - 1.0
