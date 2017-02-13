import numpy as np
import cv2
import matplotlib.pyplot as plt
from slider import half_img, histo, make_box, find_box_peak, walk_lines
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
fname = 'harder5.jpg'
original = cv2.imread('test_images/{}'.format(fname))

gEnv = {
    'debug':True,
    'left_prev_coef':None,
    'right_prev_coef':None,
    'prev_line_ends':None,
    'frame_count':0,
    'good_frames':0,
    'prev_data_good':False,
    'bad_frame_count':0
}

if gEnv['debug']:
    cv2.imshow('Original', original)

def process_frame(frame):
    global gEnv

    gEnv['frame_count'] += 1 # keep track of frames to get ratio of good ones
    # Correct for camera distortion
    corrected = correct_image(frame)

    # normalize brightness and contrast
    image = apply_CLAHE(corrected)

    # Transform it to birdseye
    image = transform2birdseye(image)
    if gEnv['debug']:
        cv2.imshow('Birdseye', image)
        cv2.imwrite('birdseye.jpg', image)
    
    # Enhance the image, and if necessary find the starts of the lines
    enhanced, gEnv['prev_line_ends'] = find_line_starts(image, gEnv)
    if gEnv['debug']:
        cv2.imshow('Enhanced', enhanced)
        cv2.imwrite('output_images/bad_bird.jpg', enhanced)

    # walk up the lines to find a series of points
    left_Xpts, right_Xpts, Ypts, data_good = walk_lines(enhanced, gEnv)

    # Fit a 2nd order function for each line
    left_coef = np.polyfit(Ypts, left_Xpts, 2)
    right_coef = np.polyfit(Ypts, right_Xpts, 2)

    # Get new lines from the fitted curve
    Yarr = np.array(Ypts)
    left_newXpts = (left_coef[0] * Yarr**2) + (left_coef[1] * Yarr) + left_coef[2]
    right_newXpts = (right_coef[0] * Yarr**2) + (right_coef[1] * Yarr) + right_coef[2]

    # was the walk data good enough to keep?
    max_bad_frames = 25  # don't reset on the first bad frame, wait a second
    if data_good:
        gEnv['bad_frame_count'] = 0
        gEnv['left_prev_coef'] = left_coef # only save coef if data was actually good, regardless of count
        gEnv['right_prev_coef'] = right_coef
        gEnv['prev_line_ends'] = [left_newXpts[len(left_newXpts) - 1], right_newXpts[len(right_newXpts) - 1]]
        gEnv['good_frames'] += 1
    else:
        gEnv['bad_frame_count'] += 1

    if gEnv['bad_frame_count'] > max_bad_frames:
        gEnv['prev_data_good'] = False
    else:
        gEnv['prev_data_good'] = True

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

    # Where are we in the lane?
    car_center = frame.shape[1] / 2 # camera is centered in the car
    left_lineX = left_newXpts[0]
    right_lineX = right_newXpts[0]
    lane_center = np.mean([left_lineX, right_lineX])
    
    birdseye_laneWpx = 906 # pixel width of lane in birdseye (points are in birdseye space)
    meters_px = 3.6576 / birdseye_laneWpx # Assumming that the lane is 12 feet wide (3.6576 meters)
    offcenter_Px = car_center - lane_center
    offcenter_Meters = offcenter_Px * meters_px
    if offcenter_Meters > 0:
        position_msg = 'Car is {:.2f} meters to the right of center'.format(abs(offcenter_Meters))
    else:
        position_msg = 'Car is {:.2f} meters to the left of center'.format(abs(offcenter_Meters))
    
    # Define conversions in x and y from pixels space to meters
    y_eval = np.max(Ypts)
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = meters_px # meters per pixel in x dimension
    left_fit_cr = np.polyfit(Yarr * ym_per_pix, np.asarray(left_Xpts) * xm_per_pix, 2)
    right_fit_cr = np.polyfit(Yarr * ym_per_pix, np.asarray(right_Xpts) * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    curvature_msg = 'Curvatures: {:.2f} meters left line, {:.2f} meters right line'.format(left_curverad, right_curverad)

    # Transform the drawn image back to normal from birdseye
    P = Perspective()
    normal = P.back2normal(warp_zero)

    # Overlay the drawn image on the corrected camera image
    result = cv2.addWeighted(corrected, 1, normal, 0.7, 0)
    cv2.putText(result, position_msg, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,0), 2)
    cv2.putText(result, curvature_msg, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,0), 2)
    if data_good:
        cv2.putText(result, 'Good Data', (10, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
    else:
        cv2.putText(result, 'Bad Data', (10, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,0), 2)

    return result


if gEnv['debug']:
    result = process_frame(original)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
else:
    from moviepy.editor import VideoFileClip

    # clip1 = VideoFileClip("challenge_video.mp4")
    clip1 = VideoFileClip("project_video.mp4")
    result_clip = clip1.fl_image(process_frame)
    # result_clip.write_videofile('challenge_video_result.mp4', audio=False)
    result_clip.write_videofile('project_video_result.mp4', audio=False)

print('{} out of {} frames were good'.format(gEnv['good_frames'], gEnv['frame_count']))