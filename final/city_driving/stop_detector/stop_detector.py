import cv2
import rospy

import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from detector import StopSignDetector

class SignDetector:
    
    def __init__(self):
        self.detector = StopSignDetector(threshold=0)
        self.publisher = rospy.Publisher("/stop_sign_distance", Float32, queue_size=10) # distance to stop sign
        self.subscriber = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.callback)
    
    def callback(self, img_msg):
        # Process image without CV Bridge
        np_img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
        bgr_img = np_img[:,:,:-1]
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # process image
        stop_sign_present, bounding_box = self.detector.predict(rgb_img)

        if not stop_sign_present:
            self.publisher.publish(1000) # no stop sign found, so say it's 1000 meters away
        else:
            x_min = bounding_box[0]
            y_min = bounding_box[1]
            x_max = bounding_box[2]
            y_max = bounding_box[3]
            sign_width = x_max - x_min
            sign_height = y_max - y_min
            sign_area = sign_width * sign_height
            if sign_area < 1: # probably not detecting right if this small
                self.publisher.publish(1000)
            else:
                dist_to_sign = self.get_distance_to_stop_sign(sign_area)
                self.publisher.publish(dist_to_sign)
        
    def get_distance_to_stop_sign(self, sign_area):
        # TODO: convert area of stop sign in pixels to distance away in meters
        return 1000

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
