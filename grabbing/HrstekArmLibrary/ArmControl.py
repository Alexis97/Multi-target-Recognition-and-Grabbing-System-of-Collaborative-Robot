#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Alexis Zhang

import os
import sys
from ctypes import *
import socket
import threading

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

# Load DLLs
# Dll_platform = 'win32'
Dll_platform = 'x64'

ControlLayerDll = WinDLL(os.path.join(curPath, Dll_platform, 'ControlLayerDll.dll')) 
libArmObj = WinDLL(os.path.join(curPath, Dll_platform, 'libArmObj.dll')) 


# Define C structure
class AngleRange(Structure):
	_fields_ = [('lowBound', c_double),
				('highBound', c_double)]

class HSArm(Structure):
	_fields_ = [('heights', c_double * 5),
				('distances', c_double * 5),
				('angles', c_double * 5),
				('g_q2anglePara', c_int),
				('controlIntervalTime', c_int),
				('strAccTime', c_int),
				('endAccTime', c_int)]

# Define RobotArm class for grasping subsystem
class RobotArm():
	def __init__(self, local_ip = 0, local_port = 10009, remote_port = 10000):
		self.local_ip = local_ip
		self.local_port = local_port
		self.remote_port = remote_port

		# Init Socket connection
		ControlLayerDll.OCU_SockInit(self.local_ip, self.local_port, self.remote_port)

	def exit(self):
		ControlLayerDll.OCU_SockExit()
		return

	def moveto(self, joints_param):
		size = 0
		ControlLayerDll.HR_ArmMoveTogether_all(joints_param, size)

	def reset(self):
		reset_param = []
		moveto(reset_param)


	def grab(self, location):
		'''
			Usage: grab a candidate object

			Parameter: 	location - bounding box location list of [x1, y1, x2, y2]

			No Return
		'''

		# Reset the robot arm
		robotArm.reset()

		# Coordinate transform
		grab_region_center_camera_view = []
		grab_region_center_robot_arm = []
		location_r = self.coordinate_transform(location, grab_region_center_camera_view, grab_region_center_robot_arm, theta)
		
		x_center = (location_r[0] + location_r[2]) / 2
		y_center = (location_r[1] + location_r[3]) / 2
		z_center = location_r[4]

		# Inverse kinematics algorithm
		joints_param = self.inverse_kinematics([x_center, y_center, z_center])

		self.moveto(joints_param)

	def coordinate_transform(self, boundingbox_c, grab_region_c, grab_region_r, theta = 1):
		'''
			Usage: Transform 2-D boundingbox information to 6 joints parameter of robot arm
			
			Parameter:	boundingbox_c - [x1, y1, x2, y2] the detection location of bounding box in Camera View coordinate
						grab_region_c - [x, y] the location of grabbing region (center) in Camera View coordinate
						grab_region_r - [x, y, z] the location of grabbing region (center) in Robot Arm coordinate
						theta - the ratio of two units, 1 (pixel from Camera View) = theta (centimeters)
						
			Return: boundingbox_r - [x1, y1, x2, y2, z] the detection location of bounding box in Robot Arm coordinate
		'''
		bx_c1, by_c1, bx_c2, by_c2 = boundingbox_c 
		gx_c, gy_c = grab_region_c
		gx_r, gy_r, gz_r = grab_region_r

		# Calculate boundingbox location in Robot Arm coordinate
		bx_r1 = gx_r + gx_c - theta * bx_c1
		by_r1 = gy_r + gy_c - theta * by_c1
		bx_r2 = gx_r + gx_c - theta * bx_c2
		by_r2 = gy_r + gy_c - theta * by_c2
		bz_r = gz_r

		return [bx_r1, by_r1, bx_r2, by_r2, bz_r]
		

	def inverse_kinematics(self, position_robot_arm):
		'''
			Usage: Call getArmAngleWithPostion in hrstekArmControl dll, to achieve inverse kinematics algorithm

			Parameter: position_robot_arm - [x, y, z] the position in Robot Arm coordinate

			Return: joints_param - list of 6 joints parameter
		'''
		armobj = libArmObj.createArmObj(armNum = 5)
		
		px, py, pz = position_robot_arm 
		qx = px
		qz = pz

		hrstekArmControl.getArmAngleWithPositon(byref(armobj), 
												px, py, pz, qx, qz, 
												byref(q0s), byref(q1s), byref(q2s), byref(q3s), byref(q4s), byref(qy))
		joints_param = [q0s, q1s, q2s, q3s, q4s]
		return joints_param


if __name__ == '__main__':
	robotArm = RobotArm()