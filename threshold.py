import numpy as np
import cv2
import matplotlib.image as mpimg
from enhancer import enhance_lines


# Read in an image
fname = 'test5.jpg'
image = mpimg.imread('test_images/{}'.format(fname))
    
# Enhance it
# enhanced = enhance_lines(image, s_thresh=(170, 255), sx_thresh=(20, 100))
# enhanced = enhance_lines(image, s_thresh=(170, 255), sx_thresh=(10, 100))
# enhanced = enhance_lines(image, HLS_thresh=[175, 250], Gray_thresh=[30, 150])
enhanced = enhance_lines(image, HLS_thresh=[175, 250], Gray_thresh=[13, 150])
cv2.imshow('Enhanced {}'.format(fname), enhanced)

# Save the result
cv2.imwrite('output_images/{}'.format(fname), enhanced)

cv2.waitKey(0)
