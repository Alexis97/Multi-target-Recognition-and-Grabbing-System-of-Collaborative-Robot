import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

import HrstekArm.hrstekArmControl 



def grab(location):
	'''
		Usage: grab a candidate object with a robot arm

		Parameter: 	location - bounding box location list of [x1, y1, x2, y2]

		No Return
	'''

	# Reset the robot arm
	robotArm = hrstekArmControl.RobotArm()
	robotArm.reset()

	# Move the arm to location and grab an object
	grab_region_center_camera_view = []
	grab_region_center_robot_arm = []
	grab_param = coordinate_transform(location, grab_region_center_camera_view, grab_region_center_robot_arm)
	
	robotArm.grab(grab_param)

def coordinate_transform(boundingbox_c, grab_region_c, grab_region_r):
	'''
		Usage: Transform 2-D boundingbox information to 6 joints parameter of robot arm
		
		Parameter:	boundingbox_c - [x1, y1, x2, y2] the detection location of bounding box in Camera View coordinate
					grab_region_c - [x, y] the location of grabbing region (center) in Camera View coordinate
					grab_region_r - [x, y, z] the location of grabbing region (center) in Robot Arm coordinate
					
		Return: list of 6 joints parameter
	'''
	bx_c1, by_c1, bx_c2, by_c2 = boundingbox_c 
	gx_c, gy_c = grab_region_c
	gx_r, gy_r, gz_r = grab_region_r

	# Calculate boundingbox location in Robot Arm coordinate
	bx_r1 = gx_r + gx_c - bx_c1
	by_r1 = gy_r + gy_c - by_c1
	bx_r2 = gx_r + gx_c - bx_c2
	by_r2 = gy_r + gy_c - by_c2
	bz_r = gz_r

	# inverse kinematics algorithm
	hrstekArmControl.getArmAngleWithPositon()

	return joints_param

