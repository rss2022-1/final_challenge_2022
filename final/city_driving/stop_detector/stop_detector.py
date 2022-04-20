import cv2
import rospy

import numpy as np
from sensor_msgs.msg import Image
from detector import StopSignDetector

class SignDetector:
    def __init__(self):
        self.detector = StopSignDetector()
        self.publisher = None #TODO: publish location of the stop sign? Just (x, y) relative to car?
        self.subscriber = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.callback)

    def callback(self, img_msg):
        # Process image without CV Bridge
        np_img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
        bgr_img = np_img[:,:,:-1]
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        #TODO: process image
        stop_sign_present, bounding_box = self.detector.predict(rgb_img)

        #TODO: retrieve coordinates in image of center of stop sign

        #TODO: convert coords to real-world and publish

if __name__=="__main__":
    rospy.init_node("stop_sign_detector")
    detect = SignDetector()
    rospy.spin()
