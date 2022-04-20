#!/usr/bin/env python

from distutils.log import error
import rospy
import numpy as np
import time

from visual_servoing.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped

class ParkingController():
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """

    rospy.set_param("parking_controller/velocity", .5)
    # PID coefficients that are all used equally and are super useful
    Kp = 0.7
    Ki = 0
    Kd = 0

    def __init__(self):
        rospy.Subscriber("/relative_cone", ConeLocation, self.relative_cone_callback)
        DRIVE_TOPIC = rospy.get_param("~drive_topic") # set in launch file; different for simulator vs racecar
        self.drive_pub = rospy.Publisher(DRIVE_TOPIC, AckermannDriveStamped, queue_size=10)
        self.error_pub = rospy.Publisher("/parking_error", ParkingError, queue_size=10)
        self.drive_cmd = AckermannDriveStamped()

        self.parking_distance = .1 # meters; try playing with this number!
        self.eps = 0.05
        self.slow_range = 0.1
        self.relative_x = 0
        self.relative_y = 0

        self.previous_error = 0
        self.integral = 0
        self.previous_time = time.time()

    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        velocity = rospy.get_param("parking_controller/velocity")

        # Correct distance from cone
        if np.abs(self.relative_x - self.parking_distance) < self.slow_range:
            # Also facing the cone
            if np.abs(self.relative_y) < self.eps:
                velocity *= (self.relative_x-self.parking_distance)/self.slow_range
                self.create_message(velocity, 0)
            # Need to adjust angle
            else:
                # Back up a bit, then re-park
                for _ in range(200):
                    error = -self.relative_y
                    output = self.pid_controller(error)
                    if output > 0:
                        angle = min(0.34, output)
                    elif output <= 0:
                        angle = max(-0.34, output)
                    self.create_message(-velocity, angle)
                    self.drive_pub.publish(self.drive_cmd)
                    self.error_publisher()
        # Cone too far in front
        elif self.relative_x - self.parking_distance > self.slow_range:
            error = self.relative_y
            output = self.pid_controller(error)
            if output > 0:
                angle = min(0.34, output)
            elif output <= 0:
                angle = max(-0.34, output)
            self.create_message(velocity, angle)
        # Cone too close
        elif self.relative_x - self.parking_distance < -self.eps:
            error = -self.relative_y
            output = self.pid_controller(error)
            if output > 0:
                angle = min(0.34, output)
            elif output <= 0:
                angle = max(-0.34, output)
            self.create_message(-velocity, angle)
        self.drive_pub.publish(self.drive_cmd)
        self.error_publisher()

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

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        # currently just straight up publishes these values
        error_msg.x_error = self.relative_x - self.parking_distance
        error_msg.y_error = self.relative_y
        error_msg.distance_error = np.sqrt(self.relative_x**2+self.relative_y**2) - self.parking_distance
        
        self.error_pub.publish(error_msg)

    def create_message(self, velocity, steering_angle):
        self.drive_cmd.header.stamp = rospy.Time.now()
        self.drive_cmd.header.frame_id = 'map'
        self.drive_cmd.drive.steering_angle = steering_angle
        self.drive_cmd.drive.steering_angle_velocity = 0
        self.drive_cmd.drive.speed = velocity
        self.drive_cmd.drive.acceleration = 0
        self.drive_cmd.drive.jerk = 0

if __name__ == '__main__':
    try:
        rospy.init_node('ParkingController', anonymous=True)
        ParkingController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
