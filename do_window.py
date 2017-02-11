import numpy as np
import cv2
import matplotlib.pyplot as plt
from slider import half_img, histo, make_box, find_box_peak
from enhancer import enhance_lines, get_enhanced, correct_image, transform2birdseye, apply_CLAHE
from perspective import Perspective
from line_start import find_line_starts

print('Import done')

"""
straight_lines1, straight_lines2: Close, but right a little inside
test1: Close, but a little narrow
test2: Very close
test3: Very close
test4: Very close
test5: White good, yellow a ways inside
test6: Vey nice
challenge1: Not too bad, a bit to the right
challenge2: Vey nice
challenge3: Not too shabby
harder1: Not bad
harder2: Not bad
harder3: Not bad, a little to the left
harder4: Not bad, a little to the right
harder5: Not bad
"""

# Read in an image
fname = 'test2.jpg'
original = cv2.imread('test_images/{}'.format(fname))
cv2.imshow('Original', original)

# Correct for camera distortion
corrected = correct_image(original)
# cv2.imwrite('corrected.jpg', corrected)

# normalize brightness and contrast
image = apply_CLAHE(corrected)

# Transform it to birdseye
image = transform2birdseye(image)
# cv2.imwrite('birdseye.jpg', image)

# find the beginning of the lines
enhanced, last_good = find_line_starts(image)
# cv2.imwrite('enhanced.jpg', enhanced)
cv2.imshow('Enhanced back', enhanced)
print('Lines start: {}'.format(last_good))
left_start = last_good[0]
right_start = last_good[1]
walk_Y = 36
half_walk = walk_Y // 2
box_width = 200
box_half = box_width // 2
curY = enhanced.shape[0]
minX = box_half # from the center of the box, don't go past the left edge of the image
maxX = enhanced.shape[1] - minX # same, right edge
curLeftX = max(left_start, minX) # don't start out already past the left edge
curRightX = min(right_start, maxX) # same, right edge
left_pts = []
right_pts = []
leftDeltaSteps = []
rightDeltaSteps = []

# Walk up the lines
while curY > walk_Y:
    left_box = make_box(enhanced, walk_Y, box_width, curY, curLeftX)
    right_box = make_box(enhanced, walk_Y, box_width, curY, curRightX)
    cv2.imshow('left_box', left_box)
    cv2.imshow('right_box', right_box)

    found_left = find_box_peak(left_box) # peak relative to box
    if found_left is not None: # did we find a peak?
        nextLeftX = max(found_left + curLeftX - box_half, minX) # don't go past the left border
        leftDeltaSteps.append(curLeftX - nextLeftX) # keep track of the deltas for averaging steps
        curLeftX = nextLeftX
    elif len(leftDeltaSteps) > 0:
        # take a step in the average direction - we can lose dashed lines if we just go straight up
        curLeftX = max(curLeftX - np.mean(leftDeltaSteps), minX)

    found_right = find_box_peak(right_box)
    if found_right is not None:
        nextRightX = min(found_right + curRightX - box_half, maxX)
        rightDeltaSteps.append(curRightX - nextRightX)
        curRightX = nextRightX
    elif len(rightDeltaSteps) > 0:
        curRightX = min(curRightX - np.mean(rightDeltaSteps), maxX)
    
    if curLeftX > minX:
        left_pts.append([curLeftX, curY + half_walk]) # Y in middle, not top edge of box
    else:
        # if the box is up against the left border, we still want to follow the line
        left_pts.append([found_left, curY + half_walk])
    
    if curRightX < maxX:
        right_pts.append([curRightX, curY + half_walk])
    else:
        right_pts.append([found_right, curY + half_walk])

    curY = curY - walk_Y
    cv2.waitKey(0)  # DEBUG

# This is where you need to fit the curvature



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
