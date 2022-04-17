#!/usr/bin/env python

import rospy
import numpy as np
import time

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
        self.wrap = 0
        self.wheelbase_length = 0.35
        self.p = .8

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
        # TODO: use hough transform to detect lanes, then send list of segments to pure pursuit
        # https://www.analyticsvidhya.com/blog/2020/05/tutorial-real-time-lane-detection-opencv/
        # https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
        raise NotImplementedError

    def pursue(self, msg):
        """
        Pursues the trajectory by steering towards the lookahead point.
        """
        lookahead_point = np.array([msg.x, msg.y])
        steering_angle = self.compute_steering_angle(lookahead_point)
        msg = self.create_ackermann_msg(steering_angle)
        self.drive_pub.publish(msg)

    def find_lookahead_point(self, lane_segments):
        """
        Finds the lookahead point based off the list of segments, by finding where the circle
        with radius lookahead distance intersects wiht line segments. Returns the average of intersected points.
        """
        # TODO: find the lookahead point by looping through all line segments and finding the intersections and
        # averaging them.
        raise NotImplementedError
        # NOTE: Old code for reference
                # Note: Only look at points further ahead on the trajectory than the
        # point returned by find_closest_point_on_trajectory
        points = np.array(self.trajectory.points[start_point_idx:])
        center = np.array(current_pose[:-1])

        # Compute the lookahead point
        intersections = []
        for i in range(min(len(points)-1, 6)):
            p1 = points[i]
            p2 = points[i+1]
            V = p2 - p1
            a = V.dot(V)
            b = 2 * V.dot(p1-center)
            c = p1.dot(p1) + center.dot(center) - 2 * p1.dot(center) - self.lookahead**2
            disc = b**2 - 4 * a * c
            if disc < 0:
                continue
            else:
                sqrt_disc = np.sqrt(disc)
                t1 = (-b + sqrt_disc) / (2 * a)
                t2 = (-b - sqrt_disc) / (2 * a)
                if 0 <= t1 <= 1 and 0 <= t2 <= 1:
                    # choose which one
                    t = max(t1, t2)
                elif 0 <= t1 <= 1:
                    t = t1
                elif 0 <= t2 <= 1:
                    t = t2
                else:
                    continue
                intersections.append(p1 + t * V)
        if len(intersections) == 0:
            # Intersection not found, how to find point to go to?
            rospy.loginfo("COULD NOT FIND INTERSECTION DO SOMETHING")
            return None
        res = intersections[-1]
        self.publish_point(res)
        return res

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


if __name__=="__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
