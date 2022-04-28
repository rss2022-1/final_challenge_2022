#!/usr/bin/env python

from cv2 import rectangle
import numpy as np
import rospy
import os

import cv2
from cv_bridge import CvBridge, CvBridgeError
import imutils

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Point32
from color_segmentation_orange import cd_color_segmentation

class ConeDetector():
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """
    def __init__(self):
        # toggle line follower vs cone parker
        # self.LineFollower = True
        # height: 376, width: 672
        # self.start_y = 250
        # self.end_y = 330

        # Subscribe to ZED camera RGB frames
        self.cone_pub = rospy.Publisher("/relative_lookahead_px", Point, queue_size=10)
        self.debug_pub = rospy.Publisher("/cone_debug_img", Image, queue_size=10)
        self.image_sub = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.image_callback)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images
        self.prev_px = (0, 0, 0)

    def image_callback(self, image_msg):
        # Apply your imported color segmentation function (cd_color_segmentation) to the image msg here
        # From your bounding box, take the center pixel on the bottom
        # (We know this pixel corresponds to a point on the ground plane)
        # publish this pixel (u, v) to the /relative_cone_px topic; the homography transformer will
        # convert it to the car frame.

        base_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        rot_image = imutils.rotate(base_image, 180)
        (h,w) = rot_image.shape[:2]

        bb, mask = cd_color_segmentation(rot_image, None)
        if bb:
            tlx, tly = bb[0] # top left
            brx, bry = bb[1] # back right
            center_x, center_y = (brx - tlx)/2.0 + tlx, bry

            cone_location = Point()
            center_img_y = h//2
            center_img_x = w//2
            thresh = 30
            if center_x < center_img_x - thresh:
                #rospy.loginfo("left")
                cone_location.x = w - tlx 
            elif center_x > center_img_x + thresh:
                #rospy.loginfo("right")
                cone_location.x = w - brx
            else:
                #rospy.loginfo("center")
                cone_location.x = w - center_x
                #cone_location.x = center_x
        
            cone_location.y = h - center_y
            #cone_location.y = center_y
            cone_location.z = 0
            self.prev_px = (cone_location.x, cone_location.y, cone_location.z)
            self.cone_pub.publish(cone_location)
            cv2.rectangle(rot_image, bb[0], bb[1], (255,255,0), 1)
            debug_msg = self.bridge.cv2_to_imgmsg(mask, "passthrough")
            self.debug_pub.publish(debug_msg)
        else:
            cone_location = Point()
            cone_location.x, cone_location.y, cone_location.z = self.prev_px
            self.cone_pub.publish(cone_location)

if __name__ == '__main__':
    try:
        rospy.init_node('ConeDetector', anonymous=True)
        ConeDetector()
        rospy.spin()
        # test_find_line()
    except rospy.ROSInterruptException:
        pass
