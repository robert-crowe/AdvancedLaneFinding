import numpy as np
import cv2
import matplotlib.pyplot as plt
from slider import half_img, histo, make_box, find_box_peak, walk_lines
from enhancer import enhance_lines, get_enhanced, correct_image, transform2birdseye, apply_CLAHE
from perspective import Perspective
from line_start import find_line_starts
from scipy.interpolate import UnivariateSpline

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
fname = 'challenge3.jpg'
original = cv2.imread('test_images/{}'.format(fname))
# cv2.imshow('Original', original)

def process_frame(frame):
    global last_good

    # Correct for camera distortion
    corrected = correct_image(frame)

    # normalize brightness and contrast
    image = apply_CLAHE(corrected)

    # Transform it to birdseye
    image = transform2birdseye(image)

    # find the beginning of the lines
    enhanced, last_good = find_line_starts(image, last_good)
    # cv2.imshow('Enhanced back', enhanced)
    # print('Lines start: {}'.format(last_good))

    # walk up the lines to find a series of points
    left_Xpts, right_Xpts, Ypts = walk_lines(last_good, enhanced)

    # Fit a 2nd order function for each line
    left_coef = np.polyfit(Ypts, left_Xpts, 2)
    right_coef = np.polyfit(Ypts, right_Xpts, 2)

    # Get new lines from the fitted curve
    Yarr = np.array(Ypts)
    left_newXpts = (left_coef[0] * Yarr**2) + (left_coef[1] * Yarr) + left_coef[2]
    right_newXpts = (right_coef[0] * Yarr**2) + (right_coef[1] * Yarr) + right_coef[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image).astype(np.uint8)

    # Form polylines and polygon
    left_newpts = [z for z in zip(left_newXpts, Ypts)]
    left_newpts = [np.array(left_newpts, dtype=np.int32)]
    right_newpts = [z for z in zip(right_newXpts, Ypts)]
    right_newpts = [np.array(right_newpts, dtype=np.int32)]
    poly_pts = np.vstack((left_newpts[0], right_newpts[0][::-1]))
    cv2.fillPoly(warp_zero, np.int_([poly_pts]), (0,255,0))

    # Draw the polylines and polygon
    cv2.fillPoly(warp_zero, np.int_([poly_pts]), (0,255,0))
    cv2.polylines(warp_zero, np.int_([left_newpts]), False, (255,0,0), thickness=30)
    cv2.polylines(warp_zero, np.int_([right_newpts]), False, (0,0,255), thickness=30)

    # Transform the drawn image back to normal from birdseye
    P = Perspective()
    normal = P.back2normal(warp_zero)

    # Overlay the drawn image on the corrected camera image
    result = cv2.addWeighted(corrected, 1, normal, 0.7, 0)

    return result


last_good = None
# result = process_frame(original)
# cv2.imshow('Result', result)
# cv2.waitKey(0)

from moviepy.editor import VideoFileClip

clip1 = VideoFileClip("project_video.mp4")
result_clip = clip1.fl_image(process_frame)
result_clip.write_videofile('project_video_result.mp4', audio=False)
