#!/usr/bin/env python

import rospy
import numpy as np
import time
import os
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from geometry_msgs.msg import Point32, Point
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32


class PurePursuit(object):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    def __init__(self):
        # Constants
        # TODO: Tune these parameters for pixel values
        self.speed = 2.0
        self.lookahead_mult = 7.0/8.0
        self.lookahead = self.lookahead_mult * self.speed
        self.px_lookahead = 230
        self.wrap = 0
        self.wheelbase_length = 0.35
        self.p = .8
        self.img_width = 640
        self.img_height = 480

        # Subscribers and publishers
        self.drive_pub = rospy.Publisher("/drive", AckermannDriveStamped, queue_size=1)
        self.mask_sub = rospy.Subscriber("/lane_segmenter/lane_mask", Image, self.mask_cb)
        self.relative_lookahead_px_pub = rospy.Publisher("/relative_lookahead_px", Point, queue_size=1)
        self.lookahead_point_sub = rospy.Subscriber("/relative_lookahead_point", Point32, self.pursue)
        self.error_pub = rospy.Publisher('/error', Float32, queue_size=1)

        rospy.loginfo("Initialized Pure Pursuit Node")

        self.current_pose = None
        self.prev_pose = None
        self.more_prev_pose = None

    def create_ackermann_msg(self, steering_angle, speed=None):
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        msg.drive.steering_angle = steering_angle
        msg.drive.steering_angle_velocity = 0.0
        msg.drive.speed = self.speed if speed is None else speed
        msg.drive.acceleration = 0.5
        msg.drive.jerk = 0.0
        return msg

    def mask_cb(self, img_msg):
        """
        Takes in an image mask and performs Hough transform to detect lanes than calls
        find_lookahead_point with the detected lane segments.
        """
        '''
        FOR COLOR SEGMENTATION:
        Options:
        - Just look out for the white lines, problem is that there is white everywhere on the track - could be mitigated by erosion/dilation and cropping
        - First segment for the track (whatever color it is) and then segment for the lanes

        FOR LINES:
        Isolate center of image to get the two lines in the center. Once we get the starting point of the lines,
        we follow the lines starting at the bottom of the image to keep track of which of the segments belong to which lane
        '''
        # TODO: use hough transform to detect lanes, then send list of segments to pure pursuit
        # https://www.analyticsvidhya.com/blog/2020/05/tutorial-real-time-lane-detection-opencv/
        # https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
        mask_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        segments = self.get_contours(mask_image)
        point_px = self.find_lookahead_point(segments)
        pt = Point()
        pt.x = point_px[0]
        pt.y = point_px[1]
        pt.z = 0
        self.relative_lookahead_px_pub.publish(pt)

    def pursue(self, msg):
        """
        Pursues the trajectory by steering towards the lookahead point.
        """
        lookahead_point = np.array([msg.x, msg.y])
        steering_angle = self.compute_steering_angle(lookahead_point)
        msg = self.create_ackermann_msg(steering_angle)
        self.drive_pub.publish(msg)

    def get_contours(self, src):
        dst = cv2.Canny(src, 50, 200, None, 3)
        # Copy edges to the images that will display the results in BGR
        cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

        linesP = cv2.HoughLinesP(dst, 1, np.pi/180, 10, None, 1, 500)

        prev_lines = [] # (theta, x1, y1, x2, y2, v)

        if linesP is not None:
            for i in range(0, len(linesP)):
                x1, y1, x2, y2 = linesP[i][0]
                v = np.array([x2-x1, y2-y1])
                th = np.arctan2(v[1], v[0])
                pixel_epsilon = 80
                scaled_v = 200 * (v / np.linalg.norm(v))
                # Delete lines that are too horizontal
                if np.abs(th) < .15:
                    continue
                # Delete lines that are similar to previous lines
                if any([(np.linalg.norm(scaled_v - prev_line[5]) < pixel_epsilon) for prev_line in prev_lines]):
                    continue
                # new_x2, new_y2 = get_intersection(x2, y2, x1, y1, w, h)
                prev_lines.append([th, x1, y1, x2, y2, scaled_v])
                cv2.line(cdstP, (x1, y1), (x2, y2), (0,0,255), 3, cv2.LINE_AA)
        return cdstP, np.array(prev_lines)

    def find_lookahead_point(self, lane_segments):
        """
        Finds the lookahead point based off the list of segments, by finding where the circle
        with radius lookahead distance intersects wiht line segments. Returns the average of intersected points.
        """
        # TODO: find the lookahead point by looping through all line segments and finding the intersections and
        # averaging them.
        if len(lane_segments) == 0:
            print("No lane segments found")
            return (0,0,0)
        lookahead_line = self.line((0, self.px_lookahead), (1000, self.px_lookahead))
        intersections = []
        for i in range(len(lane_segments)):
            l = self.line(lane_segments[i][1:3], lane_segments[i][3:5])
            x, y = self.intersection(lookahead_line, l)
            intersections.append((x, y))
        if len(intersections) == 1:
            print("One intersection found")
            x = intersections[0][0]
            if x < self.img_width / 2:
                # Left lane
                # TODO: Use homography to see how many pixels to the right we need to shift our "center"
                x += 0
            else:
                # Right lane
                # TODO: Use homography to see how many pixels to the right we need to shift our "center"
                x -= 0
            return (x, self.px_lookahead, 0)
        elif len(intersections) == 2:
            print("two intersections found")
            return (int((intersections[0][0] + intersections[1][0])/2.), self.px_lookahead, 0)
        else:
            print(str(len(intersections)) + " intersections found")
            return (0, 0, 0)

    def compute_steering_angle(self, lookahead_point):
        ''' Computes the steering angle for the robot to follow the given trajectory.
        '''
        # Compute eta - use a dot b = |a|*|b|*cos(eta) where a is our forward velocity and
        # b is the vector from the robot to the lookahead point
        x_curr, y_curr, theta_curr = 0.0, 0.0, 0.0
        x_ref, y_ref = lookahead_point
        car_vector = (np.cos(theta_curr), np.sin(theta_curr)) # direction of car
        reference_vector = (x_ref - x_curr, y_ref - y_curr) # car to reference point
        l_1 = np.linalg.norm(reference_vector)
        eta = np.arccos(np.dot(car_vector, reference_vector)/(np.linalg.norm(car_vector)*l_1))
        delta = np.arctan(2 * self.wheelbase_length * np.sin(eta) / l_1) # from lecture notes 5-6
        sign = np.sign(np.cross(car_vector, reference_vector)) # determines correct steering direction
        return sign * delta

    def line(self, p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    def intersection(self, L1, L2):
        D  = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x, y
        else:
            return 0, 0


def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	winname = "Image"
	cv2.namedWindow(winname)        # Create a named window
	cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
	cv2.imshow(winname, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def get_contours(src):
    w, h = src.shape[1], src.shape[0]
    dst = cv2.Canny(src, 50, 200, None, 3)
    # Copy edges to the images that will display the results in BGR
    cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    linesP = cv2.HoughLinesP(dst, 1, np.pi/180, 10, None, 1, 500)

    prev_lines = [] # (theta, x1, y1, x2, y2, v)

    if linesP is not None:
        for i in range(0, len(linesP)):
            x1, y1, x2, y2 = linesP[i][0]
            v = np.array([x2-x1, y2-y1])
            th = np.arctan2(v[1], v[0])
            pixel_epsilon = 80
            scaled_v = 200 * (v / np.linalg.norm(v))
            # Delete lines that are too horizontal
            if np.abs(th) < .15:
                continue
            # Delete lines that are similar to previous lines
            if any([(np.linalg.norm(scaled_v - prev_line[5]) < pixel_epsilon) for prev_line in prev_lines]):
                continue
            # new_x2, new_y2 = get_intersection(x2, y2, x1, y1, w, h)
            prev_lines.append([th, x1, y1, x2, y2, scaled_v])
            cv2.line(cdstP, (x1, y1), (x2, y2), (0,0,255), 3, cv2.LINE_AA)
    return cdstP, np.array(prev_lines)


def get_intersection(x1, y1, x2, y2, w, h):
    """
    Given a line segment returns the intersection point of the line with the rectangle with width w and height h
    when you extend the line in the direction 1 to 2
    """
    slope = float(y2 - y1) / float(x2 - x1)
    # print(slope)
    y_intercept = y1 - slope * x1
    x_intercept = -y_intercept / slope
    if x_intercept < 0: # left edge intersection
        # print("Left edge intersection")
        new_x = 0
        new_y = y_intercept
    elif x_intercept > w: # Right edge intersection
        # print("right edge intersection")
        new_x = w
        new_y = slope * w + y_intercept
    elif x_intercept > 0 and x_intercept < w: # Middle intersection
        # print("middle intersection x")
        new_x = x_intercept
        new_y = slope * x_intercept + y_intercept
    elif y_intercept < 0: # Top edge intersection
        # print("top edge intersection")
        new_x = (0 - y_intercept) / slope
        new_y = 0
    elif y_intercept > h: # Bottom edge intersection
        # print("bottom edge intersection")
        new_x = (h - y_intercept) / slope
        new_y = h
    else:
        # print("Middle intersection")
        new_x = x_intercept
        new_y = y_intercept
    return int(new_x), int(new_y)

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return 0, 0

def find_lookahead_point(lane_segments):
    if len(lane_segments) == 0:
        print("No lane segments found")
        return (0,0,0)
    px_lookahead = 230
    lookahead_line = line((0, px_lookahead), (1000, px_lookahead))
    intersections = []
    for i in range(len(lane_segments)):
        l = line(lane_segments[i][1:3], lane_segments[i][3:5])
        x, y = intersection(lookahead_line, l)
        intersections.append((x, y))
    if len(intersections) == 1:
        print("One intersection found")
        return (intersections[0][0], px_lookahead, 0)
    elif len(intersections) == 2:
        print("two intersections found")
        return (int((intersections[0][0] + intersections[1][0])/2.), px_lookahead, 0)
    else:
        print(str(len(intersections)) + " intersections found")
        return (0, 0, 0)



def test_get_lanes():
    masks_path = os.path.abspath(os.getcwd()) + "/masks/"
    for i in range(1, 24):
        src = cv2.imread(masks_path + "mask" + str(i) + ".jpg")
        image_print(src)
        cdstP, lines = get_contours(src)
        image_print(cdstP)

def test_get_intersection():
    masks_path = os.path.abspath(os.getcwd()) + "/masks/"
    for i in range(1, 24):
        src = cv2.imread(masks_path + "mask" + str(i) + ".jpg")
        cdstP, lines = get_contours(src)
        image_print(cdstP)
        lookahead_point = find_lookahead_point(lines)
        image = cv2.circle(cdstP, (lookahead_point[0], lookahead_point[1]), radius=4, color=(255, 0, 0), thickness=-1)
        print(i)
        image_print(image)







if __name__=="__main__":
    # rospy.init_node("pure_pursuit")
    # pf = PurePursuit()
    # rospy.spin()
    # test_get_lanes()
    test_get_intersection()
