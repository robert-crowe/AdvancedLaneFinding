import numpy as np
import cv2

class Perspective:
    """ Class for transforming to and from a birdseye view

    Constructor Args:
        offset (kwarg int, optional): The pixels to offset the image from the left and right margin
    """
    def __init__(self, offset=300):
        self.offset = offset # X offset for dst points
        self.source = self.getSource()
    
    def getSource(self):
        """ Get source registration points

        Returns:
            (list of 2-tuples, float32): The list of registration points, clockwise from upper left
        """
        # src = np.float32([(624,432), (654,432), (1039,674), (269,674)])
        # src = np.float32([(522,502), (766,502), (1039,674), (269,674)])
        return np.float32([(601,445), (673,445), (1022,673), (261,673)])

    def getDestination(self, img_size):
        """ Get destination registration points

        Args:
            img_size (2-tuple int): The width and height of the image        

        Returns:
            (list of 2-tuples, float32): The list of registration points, clockwise from upper left
        """
        return np.float32([[self.offset, 0], [img_size[0]-self.offset, 0], 
            [img_size[0]-self.offset, img_size[1]], [self.offset, img_size[1]]])
    
    def do_otsu(self, img):
        """ Convert RGB to grayscale and apply Otsu's transform

        Args:
            img (image): The image to be transformed.  Grayscale, or if color MUST be RGB
        
        Returns:
            otsu_gray (image): Transformed image
        """
        if img.ndim == 2:
            img = np.array(img * 255, dtype=np.uint8)
        elif (img.ndim == 3) and (img.shape[2] == 3):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        ret, otsu_gray = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return otsu_gray

    def to_birdseye(self, img, otsu=False):
        """ Transforms image from center camera to birdseye view

        This is hard coded to match the camera on the test car. 

        Args:
            img (image): The image to be transformed.  Grayscale, or if color MUST be RGB.
            otsu (kwarg bool, optional): Should we apply Otsu's transform?
        
        Returns:
            birdseye (image): The birdseye view, matching the color channels of the original
        """
        # Grab the image shape
        img_size = (img.shape[1], img.shape[0])

        # Registration points, clockwise from top left 
        src = self.source

        # Transformed registration points, clockwise from top left
        dst = self.getDestination(img_size)

        # Given src and dst points, calculate the perspective transform matrix
        Tmat = cv2.getPerspectiveTransform(src, dst)

        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, Tmat, img_size)

        if (otsu):
            warped = self.do_otsu(warped)
        
        return warped

    def back2normal(self, img, otsu=False):
        """ Transforms image from birdseye view back to normal view

        This is hard coded to match the camera on the test car. 

        Args:
            img (image): The image to be transformed.  Grayscale, or if color MUST be RGB.
            otsu (kwarg bool, optional): Should we apply Otsu's transform?
        
        Returns:
            normal_img (image): The normal view, matching the color channels of the original
        """
        # Grab the image shape
        img_size = (img.shape[1], img.shape[0])

        # Registration points, clockwise from top left 
        src = self.source

        # Transformed registration points, clockwise from top left
        dst = self.getDestination(img_size)

        # Given src and dst points, calculate the inverse perspective transform matrix
        invTmat = cv2.getPerspectiveTransform(dst, src)

        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, invTmat, img_size)

        if (otsu):
            warped = self.do_otsu(warped)
        
        return warped