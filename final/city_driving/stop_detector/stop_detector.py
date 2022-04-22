import cv2
import rospy

import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from detector import StopSignDetector

class SignDetector:
    
    def __init__(self):
        self.detector = StopSignDetector(threshold=0)
        self.publisher = rospy.Publisher("/stop_sign_distance", Float32, queue_size=10) # distance to stop sign in meters
        self.img_subscriber = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.img_callback)
        self.depth_subscriber = rospy.Subscriber("/zed/zed_node/depth/depth_registered", Image, self.depth_callback)
        self.sign_bounding_box = [0, 0, 0, 0]

    def img_callback(self, img_msg):
        # Process image without CV Bridge
        np_img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
        bgr_img = np_img[:,:,:-1]
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # process image
        stop_sign_present, bounding_box = self.detector.predict(rgb_img)

        if not stop_sign_present:
            self.sign_bounding_box = [0, 0, 0, 0]
            self.publisher.publish(1000) # no stop sign found, so say it's 1000 meters away
        else:
            self.sign_bounding_box = bounding_box
        
    def depth_callback(self, img_msg):
        depth_img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
        if self.sign_bounding_box != [0, 0, 0, 0]: # stop sign has been detected
            x_min = self.sign_bounding_box[0]
            y_min = self.sign_bounding_box[1]
            x_max = self.sign_bounding_box[2]
            y_max = self.sign_bounding_box[3]
            sign_depth_img = depth_img[y_min:y_max, x_min:x_max]
            dist = np.mean(sign_depth_img)
            self.publisher.publish(dist)

    def test_bounding_box(self, img_file_name):
        bgr_img = cv2.imread(img_file_name)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        stop_sign_present, bounding_box = self.detector.predict(rgb_img)

        if stop_sign_present:
            cv2.imshow("stop sign detector", np.array(self.detector.draw_box(bgr_img, bounding_box)))
            cv2.waitKey(0)

if __name__=="__main__":
    rospy.init_node("stop_sign_detector")
    detect = SignDetector()
    rospy.spin()
