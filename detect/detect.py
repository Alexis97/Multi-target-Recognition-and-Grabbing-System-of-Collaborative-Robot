#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Alexis Zhang

import os
import cv2
import sys
import time

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

from Server import *

def frame_detection(server, img_path, opt, timeout = 5):
	'''
		Usage: 	Send a single image file to Server to get detection result.
				Cost about 200ms to detect a frame

		Parameter: 	server - a Server class object (from file Server.py)
					img_path - the local image path for detection
					opt - argument options from __main__ function
					timeout - maximum waiting time for transmission
		
		Return: detections - list of detection bounding boxes, each detection item is [x1, y1, x2, y2, cls, conf]
	'''
	img_name = os.path.split(img_path)[1]
	local_img_path = img_path

	server_dir = opt.target_server_dir
	target_detector_dir = opt.target_detector_dir

	target_image_dir = os.path.join(server_dir, opt.local_img_dir).replace('\\', '/')			# replace '\' on windows to '/' on linux file system
	target_img_path = os.path.join(server_dir, opt.local_img_dir, img_name).replace('\\', '/')  
	target_output_dir = os.path.join(server_dir, opt.local_result_dir).replace('\\', '/')
	

	# Send local image to server
	time_0 = time.time()
	img = cv2.imread(local_img_path)
	status = server.sftp_put(local_img_path, target_img_path)
	if status == -1:
		return

	# Get detection result from server
	time_0 = time.time()
	target_output_name = os.path.splitext(img_name)[0] + '.txt'
	target_output_path = os.path.join(target_output_dir, target_output_name).replace('\\', '/')
	local_output_path = os.path.join('output', target_output_name)

	# Wait until detection over
	detection_flag = False
	loop_time_0 = time.time()
	while (time.time() - loop_time_0) < timeout:
		files = server.sftp.listdir_attr(target_output_dir)
		for file in files:
			if file.filename == target_output_name:
				status = server.sftp_get(target_output_path, local_output_path)
				detection_flag = True
				break
		if detection_flag:
			try:
				detection_file = open(local_output_path)
				detection_file.close()
				break
			except:
				detection_flag = False
				continue
	if not detection_flag:
		print ('Out of Time')
		return -1

	# return detection
	detections = []
	with open(local_output_path,'r') as detection_file:
		lines = detection_file.readlines()
		for line in lines:
			detection = line.rstrip('\n').split()
			if detection is not None:
				detection = list(map(float, detection))
				detections.append(detection)


	return detections

def plot_one_box(location, img, color=None, label=None, line_thickness=None):
	'''
		Usage: Plot one bounding box on image. This function is copyed from yolov3/utils/utils.py

		Parameter: 	location - bounding box location list of [x1, y1, x2, y2]
					img - image to be plotted
					color, line_thickness - cv2 parameters

		No Return
	'''
	tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
	color = color or [random.randint(0, 255) for _ in range(3)]
	c1, c2 = (int(location[0]), int(location[1])), (int(location[2]), int(location[3]))
	cv2.rectangle(img, c1, c2, color, thickness=tl)
	if label:
		tf = max(tl - 1, 1)  # font thickness
		t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
		c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
		cv2.rectangle(img, c1, c2, color, -1)  # filled
		cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)	
