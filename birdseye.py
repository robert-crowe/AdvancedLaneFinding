import numpy as np
import cv2
import matplotlib.image as mpimg
from perspective import Perspective


#####################################################################
# NOTE TO ME:
# IT LOOKS LIKE IT MIGHT BE A GOOD IDEA TO ADJUST THE OFFSET
# BASED ON THE CURVE.  THAN MEANS WE HAVE A DIFFERENT MATRIX
# EVERY TIME WE ADJUST.
#####################################################################



# Read in an image
fname = 'test3.jpg'
image = mpimg.imread('test_images/{}'.format(fname))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow('Original {}'.format(fname), image)
    
# Transform it
P = Perspective()
birdseye = P.to_birdseye(image, otsu=True)
cv2.imshow('Birdseye view of {}'.format(fname), birdseye)

# Save the result
# cv2.imwrite('output_images/{}'.format(fname), enhanced)

# Transform it back
normal_img = P.back2normal(birdseye)
cv2.imshow('Normal view of {}'.format(fname), normal_img)

cv2.waitKey(0)
