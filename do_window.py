import numpy as np
import cv2
import matplotlib.pyplot as plt
from slider import half_img, histo
from enhancer import enhance_lines, find_lines, get_enhanced_birdseye, correct_image, transform2birdseye, apply_CLAHE
from perspective import Perspective

#####################################################################
# NOTE TO ME:
# IT LOOKS LIKE IT MIGHT BE A GOOD IDEA TO ADJUST THE OFFSET
# BASED ON THE CURVE.  THAN MEANS WE HAVE A DIFFERENT MATRIX
# EVERY TIME WE ADJUST.
# THE STEERING ANGLE MIGHT ALSO BE A CLUE TO USE TO ADJUST THE OFFSET
#####################################################################

print('Import done')

"""
straight_lines1, straight_lines2: Good
    Very good
test1: Shifted to left
    still too wide to the left, but better
test2: Close, slightly to the left
test3: Good
test4: White good, yellow a bit left
test5: White good, yellow a bit right
test6: Pretty close, not perfect
challenge1: Pretty close, not perfect
challenge2: Didn't find lines ---------------
challenge3: Pretty close, not perfect
harder1: Pretty dang close, not perfect
harder2: Pretty dang close, not perfect
harder3: Both lines off to right, but found two lines!
harder4: Yellow good, white off to right
harder5: Yellow close, white off to right
"""


# Read in an image
fname = 'test1.jpg'
original = cv2.imread('test_images/{}'.format(fname))
cv2.imshow('Original', original)

# Correct for camera distortion
corrected = correct_image(original)

# normalize brightness and contrast
image = apply_CLAHE(corrected)

# Transform it to birdseye
image = transform2birdseye(image)

done_looking = False
window_height = image.shape[0]
min_height = 50
window = image
good_list = []
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
        good_list.append(nice_peaks)
        # Grab the lower half
        window = half_img(image, window_height)
        window_height = window.shape[0]
    else:
        done_looking = True

good_list = np.reshape(good_list, (len(good_list), 2))
last_good = [np.mean(good_list[:, 0]), np.mean(good_list[:, 1])]
print('Lines start: {}'.format(last_good))

# Let's suppose, as in the previous example, you have a warped binary 
# image called warped, and you have fit the lines with a polynomial and 
# have arrays called yvals, left_fitx and right_fitx, which represent 
# the x and y pixel values of the lines. You can then project those 
# lines onto the original image as follows:
left_fitx = np.asarray([last_good[0], last_good[0]])
right_fitx = np.asarray([last_good[1], last_good[1]])
yvals = np.array([0, image.shape[0]])

# Create an image to draw the lines on
warp_zero = np.zeros_like(image).astype(np.uint8)
# color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(warp_zero, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
# newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
P = Perspective()
normal = P.back2normal(warp_zero)

# Combine the result with the original image
result = cv2.addWeighted(corrected, 1, normal, 0.3, 0)

cv2.imshow('Result', result)

cv2.waitKey(0)
