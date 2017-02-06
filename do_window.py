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

print('Import done')

# Read in an image
fname = 'test6.jpg'

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
# birdseye = enhance_lines(birdseye, HLS_thresh=[175, 250])
birdseye = enhance_lines(birdseye, HLS_thresh=[200, 255])
cv2.imshow('Enhanced {}'.format(fname), birdseye)

# Grab the lower half
halfsies = half_img(birdseye)
cv2.imshow('Lower Half of {}'.format(fname), halfsies)
# cv2.imwrite('halfsies.jpg', halfsies)
print('halfsies shape: {}'.format(halfsies.shape))

# What does the histogram look like?
histogram = histo(halfsies)
plt.plot(histogram)
plt.show()

# Get the moving average
# avg_width = 100
# avg_width = 50
avg_width = 25
Mavg = moving_average(halfsies, width=avg_width)

# Find the nicest peaks in the moving average
typical_line_maxWpx = 50
typical_lane_maxWpx = 750
peak_thresh = 0.5
step = 0.05
num_peaks = 0
found_twins = True
nice_peaks = []
while found_twins and len(nice_peaks) is 0: 
    peaks = find_peaks_cwt(Mavg > peak_thresh, [typical_line_maxWpx], max_distances=[typical_lane_maxWpx])
    num_peaks = len(peaks)
    peak_thresh -= step
    if peak_thresh < 0 and num_peaks < 2:
        found_twins = False
    elif num_peaks > 1:
        for p1 in peaks:  # did we find a pair about the right distance apart to be a lane?
            for p2 in peaks:
                dist = np.absolute(p2 - p1)
                if dist > 600 and dist < 800:
                    nice_peaks = [p1, p2]
                    break
            if len(nice_peaks) > 0:
                break

both_lines = found_twins and len(nice_peaks) is 2

print('Peaks: {}, Nice_Peaks: {}, Both_lines: {}'.format(peaks, nice_peaks, both_lines))

plt.plot(Mavg)
plt.show()

cv2.waitKey(0)
