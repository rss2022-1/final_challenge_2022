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
      
    def get_dist_to_sign(self, depth_npy, bounding_box):
        x_min = bounding_box[0]
        y_min = bounding_box[1]
        x_max = bounding_box[2]
        y_max = bounding_box[3]
        sign_depth_npy = depth_npy[int(y_min):int(y_max), int(x_min):int(x_max)]
        dist = np.nanmean(sign_depth_npy)
        return dist
    
    def depth_callback(self, img_msg):
        depth_img = np.frombuffer(img_msg.data, dtype=np.float32).reshape(img_msg.height, img_msg.width)
        if self.sign_bounding_box != [0, 0, 0, 0]: # stop sign has been detected
            self.publisher.publish(self.get_dist_to_sign(depth_img, self.sign_bounding_box))
    
    def test_bounding_box(self, img_file_name):
        bgr_img = cv2.imread(img_file_name)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        stop_sign_present, bounding_box = self.detector.predict(rgb_img)

        if stop_sign_present:
            cv2.imshow("stop sign detector", np.array(self.detector.draw_box(bgr_img, bounding_box)))
            cv2.waitKey(0)
     
    def test_stop_sign_distance(self, rgb_img_path, depth_npy_path):
        bgr_img = cv2.imread(rgb_img_path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        cv2.imshow("rgb img", bgr_img)
        cv2.waitKey(0)

        with open(depth_npy_path, 'rb') as f:
            depth_npy = np.load(f)

        stop_sign_present, bounding_box = self.detector.predict(rgb_img)
        if stop_sign_present:
            cv2.imshow("stop sign detected rgb", np.array(self.detector.draw_box(bgr_img, bounding_box)))
            cv2.waitKey(0)
            print("distance to sign: " + str(self.get_dist_to_sign(depth_npy, bounding_box)))


if __name__=="__main__":
    rospy.init_node("stop_sign_detector")
    detect = SignDetector()
    # rgb_img_path = "../../road_detector/test_images/stopsign2/rgb/rgb46.png"
    # depth_img_path = "../../road_detector/test_images/stopsign2/depth/depth180.npy"
    # detect.test_stop_sign_distance(rgb_img_path, depth_img_path)
    rospy.spin()
