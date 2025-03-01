#!/usr/bin/env python2

import rospy
import numpy as np
import time
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Point, Point32

class CityDriver:

    Kp = 0.7
    Ki = 0
    Kd = 0

    def __init__(self):
        self.stop_sign_sub = rospy.Subscriber("/stop_sign_distance", Float32, self.stop_callback)
        #self.collision_sub = rospy.Subscriber("/collision_checker", Bool, self.collision_callback)
        self.cone_sub = rospy.Subscriber("/relative_lookahead_point", Point32, self.cone_callback)

        DRIVE_TOPIC = rospy.get_param("~drive_topic")
        rospy.loginfo(DRIVE_TOPIC)
        self.slow_speed = 0.2
        self.normal_speed = 0.5
        self.fast_speed = 1
        self.drive_pub = rospy.Publisher(DRIVE_TOPIC, AckermannDriveStamped, queue_size=10)

        self.stop_signal = 0
        self.collision_signal = 0
        self.drive_message = AckermannDriveStamped()

        self.parking_distance = 0.1
        # self.eps = 0.05
        self.previous_error = 0
        self.integral = 0
        self.steering_angle = 0
        self.previous_time = time.time() # For PID controller
        self.stopped_time = time.time() # Measures time to stop at sign
        self.backup_time = time.time() # Measures time to back up

    def stop_callback(self, msg):
        """
        Callback for detecting stop signs.
        Based on distance, either keep going, slow down, stop, or resume driving
        Modifies class variable self.stop_signal:
            0: Keep going (normal conditions)
            1: Slow down (approaching stop sign)
            2: Stop (within stopping distance + desired stop time hasn't yet expired)
            3: Resume driving (can still see stop sign)
        """

        distance = msg.data
        
        # Stops for 1 second at sign before resuming
        if self.stop_signal == 2:
            curr_time = time.time()
            if curr_time - self.stopped_time > 1:
                self.stop_signal = 3
        else:
            if distance > 5:
                self.stop_signal = 0
            elif (distance > 0.9 or distance < 5):
                self.stop_signal = 1
            elif (distance > 0.75 or distance < 0.9):
                if self.stop_signal != 3:
                    self.stop_signal = 2
                    self.stopped_time = time.time()
            elif distance < 0.75:
                self.stop_signal = 3

    def collision_callback(self, msg):
        """
        Based on checker + scans, stop and back up or do nothing
        Modifies class variable self.collision_signal:
            0: Do nothing
            1: Stop + back up
        """
        colliding = msg.data

        # Gives time for car to backup
        if self.collision_signal == 1:
            curr_time = time.time()
            if curr_time - self.backup_time > 0.5:
                self.collision_signal = 0
        # Sets collision signal according to scan data
        else:
            if colliding == True:
                self.collision_signal = 1
                self.backup_time = time.time()
            else:
                self.collision_signal = 0

    def cone_callback(self, msg):
        """
        Based on relative cone position, create a drive message
        Modifies class variable self.steering_angle
        """
        relative_x = msg.x
        relative_y = msg.y
        velocity = self.normal_speed
        #rospy.loginfo(msg)
        #rospy.loginfo("got cone msg")

        # Cone too far in front
        if relative_x - self.parking_distance > self.parking_distance:
            #rospy.loginfo("go forward")
            error = relative_y
            output = self.pid_controller(error)
            if output > 0:
                angle = min(0.34, output)
            elif output <= 0:
                angle = max(-0.34, output)
            angle = self.compute_steering_angle(relative_x+.3, relative_y)
            self.steering_angle = angle
        # Cone too close
        # Do we even need this part? Needed for parking controller but prob not line follower
        self.drive_controller()

    def pid_controller(self, error):
        curr_time = time.time()
        dt = curr_time - self.previous_time
        prop = error
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.Kp * prop + self.Ki * self.integral + self.Kd * derivative
        # Reset previous error/time values
        self.previous_error = error
        self.previous_time = curr_time
        return output

    def compute_steering_angle(self, x, y):
        x_curr, y_curr, theta_curr = 0.0, 0.0, 0.0
        car_vector = (np.cos(theta_curr), np.sin(theta_curr))
        reference_vector = (x - x_curr, y - y_curr)
        l_1 = np.linalg.norm(reference_vector)
        eta = np.arccos(np.dot(car_vector, reference_vector)/(np.linalg.norm(car_vector)*l_1))
        delta = np.arctan(2*.35*np.sin(eta)/l_1)
        sign = np.sign(np.cross(car_vector, reference_vector))
        return sign * delta

    def drive_controller(self):
        """
        Master logic for city driving
        """
        # Priority 1: Collision checking
        # Back straight up if about to collide
        if self.collision_signal == 1:
            self.create_message(-0.2,0)
            self.drive_pub.publish(self.drive_message)
        # Priority 2: Stop sign detection
        else:
            # Keep going
            if self.stop_signal == 0:
                #rospy.loginfo("driving")
                self.create_message(self.normal_speed, self.steering_angle)
                self.drive_pub.publish(self.drive_message)
            # Slow down
            elif self.stop_signal == 1:
                self.create_message(self.slow_speed, self.steering_angle)
                self.drive_pub.publish(self.drive_message)
            # Stop
            elif self.stop_signal == 2:
                self.create_message(0, 0)
                self.drive_pub.publish(self.drive_message)
            # Keep driving slowly
            elif self.stop_signal == 3:
                self.create_message(self.slow_speed, self.steering_angle)
                self.drive_pub.publish(self.drive_message)


    def create_message(self, velocity, steering_angle):
        self.drive_message.header.stamp = rospy.Time.now()
        self.drive_message.header.frame_id = 'map'
        self.drive_message.drive.steering_angle = steering_angle
        self.drive_message.drive.steering_angle_velocity = 0
        self.drive_message.drive.speed = velocity
        self.drive_message.drive.acceleration = 0
        self.drive_message.drive.jerk = 0

if __name__=="__main__":
    rospy.init_node("city_driver")
    driver = CityDriver()
    rospy.spin()

    
