import numpy as np
import cv2
import matplotlib.pyplot as plt
from slider import half_img, histo
from enhancer import enhance_lines, find_lines, get_enhanced_birdseye, correct_image, transform2birdseye, apply_CLAHE

#####################################################################
# NOTE TO ME:
# IT LOOKS LIKE IT MIGHT BE A GOOD IDEA TO ADJUST THE OFFSET
# BASED ON THE CURVE.  THAN MEANS WE HAVE A DIFFERENT MATRIX
# EVERY TIME WE ADJUST.
#####################################################################

print('Import done')

# Read in an image
fname = 'test1.jpg'
image = cv2.imread('test_images/{}'.format(fname))
cv2.imshow('Original', image)

# Correct for camera distortion
image = correct_image(image)

# normalize brightness and contrast
image = apply_CLAHE(image)
cv2.imshow('CLAHE', image)

# Transform it to birdseye
image = transform2birdseye(image)

done_looking = False
window_height = image.shape[0]
min_height = 50
window = image
last_good = []
while not done_looking and window_height > min_height:
    # get the enhanced birdseye view
    window = get_enhanced_birdseye(window)

    # What does the histogram look like?
    histogram = histo(window)
    plt.plot(histogram)
    plt.show()

    # Try to find two good lines
    peaks, nice_peaks, both_lines = find_lines(window)

    if both_lines:
        print('Peaks: {}, Nice_Peaks: {}, Both_lines: {}'.format(peaks, nice_peaks, both_lines))
        last_good = nice_peaks
        # Grab the lower half
        window = half_img(image, window_height)
        window_height = window.shape[0]
    else:
        done_looking = True

print('Lines start: {}'.format(last_good))

cv2.waitKey(0)
