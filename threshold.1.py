import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from sobel import abs_sobel_thresh


# Read in an image
fname = 'test5.jpg'
image = mpimg.imread('test_images/{}'.format(fname))
    
# Transform it
# grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=0, thresh_max=255)
cv2.imshow('Sobel {}'.format(fname), grad_binary)

# Save the result
cv2.imwrite('output_images/{}'.format(fname), grad_binary)

cv2.waitKey(0)
