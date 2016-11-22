#!/usr/bin/env python

import os, sys
import simplejson as json
import time

import cv2
from pylab import *
import rospy
import numpy
from std_msgs.msg import Float32MultiArray, Int16, Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
import tf


class angle_data():
    def __init__(self):
        self.theta = None
        self.sonar = None
        self.image = None
        self.temperture = None
        self.odom_x = None
        self.odom_y = None
        self.odom_z = None

    # def printToFile(self, fd, counter):
    #     fd.write("--------------------------------------------- \n")
    #     fd.write("time stamp: " + str(rospy.get_rostime()) + "\n")
    #     fd.write("theta: " + str(self.theta) + "\n")
    #     fd.write("Image: " + self.image + "\n")
    #     fd.write("Sonar: " + self.sonar + "\n")
    #     fd.write("odom: \n")
    #     fd.write(
    #         "\t x: " + str(self.odom_x) + "\t y: " + str(self.odom_y) + "\t phi(oriantion): " + str(self.odom_z) + "\n")
    #     if (self.theta == -180):
    #         fd.write("**************************************************\n")

    def createDictForCapture(self):
        diction = {}
        diction['angle'] = self.theta
        diction['capture'] = {'Image': self.image, 'Sonar': self.sonar, 'Temperture': self.temperture,'odom_x': self.odom_x,
                                             'odom_y': self.odom_y, 'phi': self.odom_z, 'time': time.time()}
        return diction


def callback_theta(data):
    global flag_theta, struct_buffer, start
    struct_buffer.append(angle_data())
    struct_buffer[flag_theta].theta = data.data
    flag_theta += 1
    start = True


def callback_image(data):
    global flag_image, struct_buffer, name, start
    bridge = CvBridge()
    msg = bridge.imgmsg_to_cv2(data, "bgr8")
    namePic = name + '/p' + str(flag_image) + '.jpg'
    cv2.imwrite(namePic, msg)
    namePic = '/p' + str(flag_image) + '.jpg'
    struct_buffer[flag_image].image = namePic
    flag_image += 1
    start = True


def callback_sonar(data):
    global flag_sonar, struct_buffer, name, start
    data = data.data
    nameSonar = '/s' + str(flag_sonar) + '.csv'
    sonarfile = open(name + nameSonar, 'w')
    numpy.savetxt(sonarfile, data)
    sonarfile.close()
    struct_buffer[flag_sonar].sonar = nameSonar
    flag_sonar += 1
    start = True

def callback_temperture(data):
    global struct_buffer, flag_tempeture
    struct_buffer[flag_tempeture].temperture = data.data
    flag_tempeture += 1
    start = True



def callback_odom(data):
    global x, y, z
    x = data.pose.pose.position.x
    y = data.pose.pose.position.y
    (r, p, z) = tf.transformations.euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
    z *= 180/3.14


def add_odom(pos):
    global x, y, z, struct_buffer
    struct_buffer[pos].odom_x = x
    struct_buffer[pos].odom_y = y
    struct_buffer[pos].odom_z = z


def exp_org():
    global struct_buffer, flag_sonar, flag_image, flag_theta, start, name, x, y, z, flag_tempeture
    x = 0
    y = 0
    z = 0
    dirname = sys.argv[1]
    name = '/home/shaul/bagfiles/' + dirname
    try:
        os.mkdir(name)
    except:
        print "Directory exist, Change the directory name"
        exit()
    # fo = os.open(name + '/metadata.txt', os.O_RDWR | os.O_CREAT)
    # fd = os.fdopen(fo, 'w+')
    # fd.write("Start \n")
    # fd.write("**************************************************\n")
    sub_sonar = rospy.Subscriber('/Sonar_file', Float32MultiArray, callback_sonar)
    sub_pic = rospy.Subscriber('/image', Image, callback_image)
    sub_theta = rospy.Subscriber('/theta', Int16, callback_theta)
    sub_odom = rospy.Subscriber("/odom", Odometry, callback_odom)
    sub_temp = rospy.Subscriber("/temperture", Float64, callback_temperture)
    rospy.init_node('exp_organization', anonymous=False)
    rospy.loginfo("Waiting for data")
    rate = rospy.Rate(10)  # 10hz
    start = False
    flag_sonar = 0
    flag_image = 0
    flag_theta = 0
    flag_tempeture = 0
    counter = 0
    scan_counter = 0
    struct_buffer = []
    dictBuffer = []
    while not rospy.is_shutdown():
        if start:
            if (struct_buffer[counter].theta != None) and (struct_buffer[counter].sonar != None) and (struct_buffer[counter].image != None) and (struct_buffer[counter].temperture != None):
                add_odom(counter)
                dictBuffer.append(struct_buffer[counter].createDictForCapture())
                if struct_buffer[counter].theta == -179:
                    scan_counter += 1
                    jsonName = "/scan%s.json" % scan_counter
                    obj = open(name + jsonName, 'wb')
                    json.dump(dictBuffer, obj)
                    obj.close()
                    dictBuffer = []
                counter += 1
                start = False
        rate.sleep()
    # fd.close()
    obj.close()


if __name__ == "__main__":
    try:
        exp_org()
    except rospy.ROSInterruptException:
        pass

