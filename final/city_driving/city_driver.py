#!/usr/bin/env python2

import rospy
import numpy as np
from std_msgs.msg import Float32

class CityDriver:
    def __init__(self):
        self.stop_sign_sub = rospy.Subscriber("/stop_sign_distance", Float32, self.stop_cb)
        self.stop = False 

        # any other important subscribers go here
        # what about line follower?

    def stop_cb(self, msg):
        """
        Callback for detecting stop signs.
        """
        # if distance < 1 meter set self.stop = True
        # else self.stop = False

        pass

if __name__=="__main__":
    rospy.init_node("city_driver")
    driver = CityDriver()
    rospy.spin()

    
