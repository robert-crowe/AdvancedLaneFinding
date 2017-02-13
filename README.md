# Advanced Lane Finding
A more advanced computer vision approach to finding lane lines with a forward facing camera.

February 13, 2017

Robert Crowe

The goals / steps of this project are the following:
* Compute the camera calibration matrix and distortion coefﬁcients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and ﬁt to ﬁnd the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Camera Calibration

I calibrate the camera matrix and distortion coefficients in calibrate_camera.py and save the result to a pickle file.  The function collect_points() uses cv2.findChessboardCorners() to collect all the object and image points from all calibration images.  I then run a test using the matrix and coefficients with cv2.undistort() and save the result, then reload the pickle file and run a second test, comparing the results to make sure that the pickle is good.  Mmmm, pickley.
The correction is subtle, but notice that the camera lens introduces curves in the lines, which should be straight.  The correction corrects that.

# Pipeline (Single Images)

My pipeline runs in process_frame(), which is in main.py and uses correct_image() from enhancer.py to load the pickle and correct the image.
 
Through experimentation I evolved my sequence to do this a little differently.  First I apply a CLAHE (Contrast Limited Adaptive Histogram Equalization) correction to adjust contrast and brightness.

Then I transform the corrected image to a birds eye view (details below).  I then use the birds eye view to run the transform to a thresholded binary image:
 
The transform to a binary thresholded image is in enhance_lines() in enhancer.py.  After converting from RGB to HSV I apply color thresholds, determined through experimentation, to locate both yellow and white lines.  I found that for white I needed at least two sets of thresholds, because white was so often present in the background, resulting in a very noisy image.  I use the sum of the pixels in the color thresholded image to detect when I have a noisy image.  Lines should only set a limited maximum number of pixels, so if I have too many set then the sum of the image pixels will be too high, indicating that there are pixels that are not part of the lines.
Originally I experimented with thresholding in HLS space, and with the Sobel transform and gradients from the code exercises (see sobel.py), but I found that I was able to achieve good results with color thresholding alone and reduce the processing requirements.  Since I was hoping to process frames in near real time, as a car will have to, I wanted to minimize processing.

I created a class Perspective in perspective.py to handle the transformations to and from the birds eye view (see images above).  I chose the calibration points largely through experimentation, knowing that the points should form a rectangle in real space on the road, and that straight lines on the road should result in straight lines when viewed from above.  I also wanted some margin on the left a right to allow for curved lines, which otherwise exit the image before reaching the top.

I first find the lines closest to the car by using find_line_starts() in line_start.py when I don’t have them already from a previous frame.  If I already have the lines from a previous frame, and the data looks good, I use those to avoid the additional processing of finding them again.
find_line_starts() does a walk down the image, starting with the full image and taking the bottom half at each iteration until it reaches a minimum size.  By starting with the whole image we account for gaps in dashed lines, but as we walk down we account for line curvature towards the top of the image.  If we can find the lines in the last slice closest to the car, they are likely to be the most accurate, so those pixels are included in all iterations.
I then use sliding windows to walk up the left and right lines, locating points along the lines as I go.  See walk_lines() in slider.py.  Both walk_lines() and find_line_starts() use a moving average of the pixels and look for peaks in the signal using scipy.signal.find_peaks_cwt().  When I can’t find a peak, for example during the gaps in dashed lines, I use the coefficients from the previous good data to calculate where the lines should be, or use a mean of the current line direction when I don’t have coefficients.
Using numpy.polyfit() I fit a 2nd order polynomial to the points that are found with the sliding window, and then use the coefficients to generate a new, smoother line.

I use the X positions of the lines and the car center (which is also the image center) to determine the position of the car in the lane.  I estimate the meters per pixel in real space and use numpy.polyfit() again to calculate the curvature of the lines in real space, since the previous numpy.polyfit() was using pixels in birds eye space.  I add the car position and curvatures along with the lines and lane overlay to the image before returning it.

# Pipeline (video)

It works well on project_video.mp4, see project_video_result.mp4

It has issues on both challenge_video.mp4 and harder_challenge_video.mp4.  I tested the single image version with frames from both of these videos to try to move closer to a version that would work with the videos, but that is still a work in progress.
