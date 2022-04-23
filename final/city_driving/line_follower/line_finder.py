#!/usr/bin/env python

import numpy as np
import rospy

from cv_bridge import CvBridge
import imutils

from sensor_msgs.msg import Image

# import your color segmentation algorithm; call this function in ros_image_callback!
from color_segmentation import cd_color_segmentation


class LineFinder():
    """
    A class for segmenting an image based on color to find a line.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image): the live RGB image from the onboard camera.
    Publishes to: /line_segmenter/line_mask (Image): a binary image showing the line mask.
    """

    def __init__(self):
        self.image_sub = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.image_callback)
        self.mask_pub = rospy.Publisher("/line_segmenter/line_mask", Image, queue_size=1)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

    def image_callback(self, image_msg):
        """
        Applies our color segmentation algorithm to the image msg and publishes
        the resulting mask.
        """

        base_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        rot_image = imutils.rotate(base_image, 180)
        # TODO: Crop the image to only look at a certain part of the frame where the lines will be
        # https://www.analyticsvidhya.com/blog/2020/05/tutorial-real-time-lane-detection-opencv/
        # See ^ for examples of cropping and doing cv2 stuff
        mask = cd_color_segmentation(rot_image)
        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, "mono8"))

if __name__ == '__main__':
    try:
        rospy.init_node('LineFinder', anonymous=True)
        LineFinder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
