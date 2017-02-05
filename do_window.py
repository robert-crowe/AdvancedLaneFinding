import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from perspective import Perspective
from slider import half_img, histo, moving_average
from scipy.signal import find_peaks_cwt
from enhancer import enhance_lines


#####################################################################
# NOTE TO ME:
# IT LOOKS LIKE IT MIGHT BE A GOOD IDEA TO ADJUST THE OFFSET
# BASED ON THE CURVE.  THAN MEANS WE HAVE A DIFFERENT MATRIX
# EVERY TIME WE ADJUST.
#####################################################################



# Read in an image
fname = 'test1.jpg'

# Reload the pickle
dist_unpickled = pickle.load(open("camera_pickle.p", "rb"))
mtx = dist_unpickled["mtx"]
dist = dist_unpickled["dist"]

# Correct the camera
image = cv2.imread('test_images/{}'.format(fname))
image = cv2.undistort(image, mtx, dist, None, mtx)
cv2.imshow('Original {}'.format(fname), image)

# Transform it to birdseye
P = Perspective()
birdseye = P.to_birdseye(image)
cv2.imshow('Birdseye view of {}'.format(fname), birdseye)
print('birdseye shape: {}'.format(birdseye.shape))

# Enhance the lines
birdseye = enhance_lines(birdseye, HLS_thresh=[175, 250], Gray_thresh=[13, 150])
cv2.imshow('Enhanced {}'.format(fname), birdseye)

# Grab the lower half
halfsies = half_img(birdseye)
cv2.imshow('Half of {}'.format(fname), halfsies)
# cv2.imwrite('halfsies.jpg', halfsies)
print('halfsies shape: {}'.format(halfsies.shape))

# What does the histogram look like?
histogram = histo(halfsies)
plt.plot(histogram)
plt.show()

# Get the moving average
avg_width = 100
Mavg = moving_average(halfsies, width=avg_width)

# Find the strongest peaks in the moving average
typical_line_maxWpx = 50
typical_lane_maxWpx = 750
peak_thresh = 0.5
step = 0.05
num_peaks = 0
while num_peaks < 2: 
    peaks = find_peaks_cwt(Mavg > peak_thresh, [typical_line_maxWpx], max_distances=[typical_lane_maxWpx])
    num_peaks = len(peaks)
    peak_thresh -= step

print('Peaks: {}'.format(peaks))

plt.plot(Mavg)
plt.show()

cv2.waitKey(0)
