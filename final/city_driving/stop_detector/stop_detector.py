import cv2
import rospy

import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from detector import StopSignDetector

class SignDetector:
    
    def __init__(self):
        self.use_depth = True
        self.use_online_detector = True
        if self.use_online_detector:
            self.box_subscriber = rospy.Subscriber("stop_sign_bbox", Float32MultiArray, self.box_callback)
        else:
            self.img_subscriber = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.img_callback)
        
        self.depth_subscriber = rospy.Subscriber("/zed/zed_node/depth/depth_registered", Image, self.depth_callback)
        self.detector = StopSignDetector(threshold=0)
        self.publisher = rospy.Publisher("/stop_sign_distance", Float32, queue_size=10) # distance to stop sign in meters
        self.sign_bounding_box = [0, 0, 0, 0]
        self.dist_measurements = [0.8636, 1.1938, 1.6256] # distances away in meters stop sign images taken on bot
        self.area_measurements = [7246.09, 3515.56, 1856.76] # area of stop sign bounding box in pixels from stop sign images
        
    
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
        depth_img = np.frombuffer(img_msg.data, dtype=np.float32).reshape(img_msg.height, img_msg.width)
        if self.sign_bounding_box != [0, 0, 0, 0]: # stop sign has been detected
            if self.use_depth:
                self.publisher.publish(self.get_dist_to_sign_from_depth(depth_img, self.sign_bounding_box))
            else:
                self.publisher.publish(self.get_dist_to_sign_from_area(self.sign_bounding_box))
    
    def box_callback(self, box_msg):
        self.sign_bounding_box = box_msg.data

    def get_coords_from_box(self, bounding_box):
        """
        Given the bounding box of a stop sign, returns x_min, y_min, x_max, y_max coords
        """
        return bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]

    def get_dist_to_sign_from_depth(self, depth_npy, bounding_box):
        """
        Given the depth information from the depth camera and the bounding box of the
        stop sign, returns the estimated distance to the stop sign in meters
        """
        x_min, y_min, x_max, y_max = self.get_coords_from_box(bounding_box)
        sign_depth_npy = depth_npy[int(y_min):int(y_max), int(x_min):int(x_max)]
        dist = np.nanmean(sign_depth_npy)
        return dist

    def get_dist_to_sign_from_area(self, bounding_box):
        """
        Given the area of the bounding box of the stop sign, returns the estimated distance 
        to the stop sign in meters
        """
        params = np.polyfit(self.area_measurements, self.dist_measurements, 2)
        a = params[0]
        b = params[1]
        c = params[2]
        area = self.get_area_bounding_box(bounding_box)
        return a*(area**2) + b*area + c

    def get_area_bounding_box(self, bounding_box):
        """
        Returns in the area of the bounding box of the stop sign in pixels^2
        """
        x_min, y_min, x_max, y_max = self.get_coords_from_box(bounding_box)
        return (x_max - x_min)*(y_max - y_min)

    def test_bounding_box(self, img_file_name):
        bgr_img = cv2.imread(img_file_name)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        stop_sign_present, bounding_box = self.detector.predict(rgb_img)

        if stop_sign_present:
            cv2.imshow("stop sign detector", np.array(self.detector.draw_box(bgr_img, bounding_box)))
            cv2.waitKey(0)
        
        print("bounding box area: " + str(self.get_area_bounding_box(bounding_box)))
     
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
            if self.use_depth:
                print("distance to sign: " + str(self.get_dist_to_sign_from_depth(depth_npy, bounding_box)))
            else:
                print("distance to sign: " + str(self.get_dist_to_sign_from_area(bounding_box)))


if __name__=="__main__":
    rospy.init_node("stop_sign_detector")
    detect = SignDetector()
    # rgb_img_path = "../../road_detector/test_images/stopsign2/rgb/rgb45.png"
    # detect.test_bounding_box(rgb_img_path)
    # depth_img_path = "../../road_detector/test_images/stopsign2/depth/depth180.npy"
    # detect.test_stop_sign_distance(rgb_img_path, depth_img_path)
    rospy.spin()
