#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Alexis Zhang

import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

from HrstekArmLibrary import ArmControl

def grab(robotArm, location):
	'''
		Usage: grab a candidate object

		Parameter: 	robotArm - a robot arm object of RobotArm class, defination in HrstekArmLibrary/ArmControl.py
					location - bounding box location list of [x1, y1, x2, y2]

		No Return
	'''
	# Reset the robot arm
	robotArm.reset()

	# Coordinate transform
	grab_region_center_camera_view = []
	grab_region_center_robot_arm = []
	location_r = robotArm.coordinate_transform(location, grab_region_center_camera_view, grab_region_center_robot_arm, theta)
	
	x_center = (location_r[0] + location_r[2]) / 2
	y_center = (location_r[1] + location_r[3]) / 2
	z_center = location_r[4]

	# Inverse kinematics algorithm
	joints_param = robotArm.inverse_kinematics([x_center, y_center, z_center])

	# Execute grasp order
	robotArm.moveto(joints_param)



