#!/usr/bin/env python2
import numpy as np
import math
import time
import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from ackermann_msgs.msg import AckermannDriveStamped

class CollisionChecker:
    SCAN_TOPIC = "/scan"
    DRIVE_TOPIC = "/vesc/high_level/ackermann_cmd_mux/output"
    SAFETY_TOPIC = "/collision_checker" 
    rospy.set_param("collision_checker/time_to_crash", 0.5)
    rospy.set_param("collision_checker/car_width", 0.17)

    LIDAR_OFFSET = np.pi/3. # (55. * np.pi / 180.)
    NUM_LIDAR_SCANS = 897

    def __init__(self):
        rospy.Subscriber(self.DRIVE_TOPIC,AckermannDriveStamped, self.drive_callback)
        rospy.Subscriber(self.SCAN_TOPIC, LaserScan, self.scan_callback)

        self.safety_pub = rospy.Publisher(self.SAFETY_TOPIC, Bool, queue_size=10)
        self.current_ranges = []  # Current laser scan ranges
        self.current_drive = AckermannDriveStamped()  # Current Drive message
        self.current_angles = []  # Current laser scan angles
        self.scan = None
        self.flag = False
        self.previous_scans = None
        self.full_scan = np.array([-1. for i in range(self.NUM_LIDAR_SCANS)])

    def goingToCrash(self, min_dist):
        """
            Checks if da car boutta crash in next self.time_to_crash seconds
            Return: True if boutta crash, False otherwise
        """
        steering_angle = self.current_drive.drive.steering_angle
        car_width = rospy.get_param("collision_checker/car_width")

        # Get angles in front of car
        angles = [(idx, val - self.LIDAR_OFFSET) for idx,val in enumerate(self.current_angles) if val > -np.pi/2 + self.LIDAR_OFFSET and val < np.pi/2 + self.LIDAR_OFFSET]

        # for i, theta in angles:
        #     if self.full_scan[i] < min_dist:
        #         return True
        # return False

        #Going straight
        if steering_angle == 0:
            theta_star = np.arctan(car_width/min_dist)
            for i, theta in angles:
                closest_object = self.full_scan[i]
                if closest_object > 10:
                    return False
                if (theta >= -np.pi/2) and (theta < -theta_star):
                    gamma = np.pi/2 + theta
                    r = car_width / np.cos(gamma)
                    if closest_object < r:
                        return True
                elif (theta >= -theta_star) and (theta < theta_star):
                    r = min_dist / np.cos(theta)
                    if closest_object < r:
                        return True
                elif (theta >= theta_star) and (theta <= np.pi/2):
                    gamma = np.pi/2 - theta
                    r = car_width / np.cos(gamma)
                    if closest_object < r:
                        return True
        # Turning left
        elif steering_angle > 0:
            theta_star = np.arctan(car_width/min_dist)
            x = min_dist * np.abs(np.tan(steering_angle))
            theta_prime = np.arctan(x + car_width)
            for i, theta in angles:
                closest_object = self.full_scan[i]
                if closest_object > 10:
                    return False
                if (theta >= -np.pi/2) and (theta < -theta_star):
                    gamma = np.pi/2 + theta
                    r = car_width / np.cos(gamma)
                    if closest_object < r:
                        return True
                elif (theta >= -theta_star) and (theta < theta_prime):
                    r = min_dist / np.cos(theta)
                    if closest_object < r:
                        return True
                elif (theta >= theta_prime) and (theta <= np.pi/2):
                    b = np.pi/4 + steering_angle
                    c = theta - steering_angle
                    r = np.sin(b) / np.sin(c) * car_width
                    if closest_object < r:
                        return True
        # Turning right
        elif steering_angle < 0:
            theta_star = np.arctan(car_width/min_dist)
            x = min_dist * np.abs(np.tan(steering_angle))
            theta_prime = np.arctan(x + car_width)
            for i, theta in angles:
                closest_object = self.full_scan[i]
                if closest_object > 10:
                    return False
                if (theta >= -np.pi/2) and (theta < -theta_prime):
                    b = np.pi/4 - steering_angle
                    c = steering_angle - theta
                    r = np.sin(b) / np.sin(c) * car_width
                    if closest_object < r:
                        return True
                elif (theta >= -theta_prime) and (theta < theta_star):
                    r = min_dist / np.cos(theta)
                    if closest_object < r:
                        return True
                elif (theta >= theta_star) and (theta <= np.pi/2):
                    gamma = np.pi/2 - theta
                    r = car_width / np.cos(gamma)
                    if closest_object < r:
                        return True

    def dont(self):
        """
            Stops the car... in style
        """
        self.safety_pub.publish(True)

    def scan_callback(self, scan):
        self.scan = scan
        self.current_ranges = np.array(scan.ranges)
        self.current_angles = np.array([scan.angle_min + i*scan.angle_increment for i in range(self.NUM_LIDAR_SCANS)])

        if not self.flag:
            self.previous_scans = self.current_ranges
            self.flag = True
            return
        else:
            for i in range(self.NUM_LIDAR_SCANS):
                self.full_scan[i] = min(self.previous_scans[i], self.current_ranges[i])/2.0
            self.previous_scans = None
            self.flag = False

        min_dist = rospy.get_param("collision_checker/time_to_crash") * self.current_drive.drive.speed
        if self.goingToCrash(min_dist):
            self.dont()

    def drive_callback(self, data):
        self.current_drive = data 

if __name__ == "__main__":
    rospy.init_node('collision_checker')
    collision_checker = CollisionChecker()
    rospy.spin()
