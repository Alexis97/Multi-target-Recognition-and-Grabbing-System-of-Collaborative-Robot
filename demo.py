#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:AlexisZhang

import os
import sys
import cv2
import threading
from queue import Queue
import time
import argparse
import paramiko
import copy
from PIL import Image

from detect.detect import *
from detect.Server import *
from grabbing.grab import *
from grabbing.HrstekArmLibrary import ArmControl

detections = []
detection_lock = threading.Lock()
detection_use = False

class DetectionThread(threading.Thread):
	'''
		Thread for object detection.
		Call function 'frame_detection' from detect/detect.py to send an image to GPU server and wait for detection result. 
	'''
	def __init__(self, server, opt):
		super().__init__()
		self.server = server
		self.opt = opt
		save_frame_name = '1.jpg'
		self.save_frame_path = os.path.join(self.opt.local_img_dir, save_frame_name)

	def run(self):
		global detections, detection_lock, detection_use

		# Loop doing detection
		while True:
			if detection_lock.acquire():
				detection_use = True
				time_0 = time.time()
				detections = frame_detection(self.server, self.save_frame_path, self.opt)
				if detections == -1:
					detection_lock.release()
					continue
				if self.opt.print_debug_info:
					print ('Detect Time Cost: %.3fs' % (time.time() - time_0))
				detection_lock.release()
				time.sleep(0.05)

		# print ('------------detection end----------------')

class GUIThread(threading.Thread):
	'''
		Thread for GUI display.
		Base on OpenCV library.
		Call function 'mouse_event' to highlight focus object, and detect the click event from user to send boundingbox info to grab subsystem.
	'''
	def __init__(self, server, opt, robotArm = None):
		super().__init__()
		self.server = server
		self.opt = opt
		self.robotArm = robotArm

		self.counter = 0
		# self.frame = None
		self.color = [0, 255, 0]
		self.color_activate = [255, 0, 0] 
		self.activate_id = -1

		# Read label list
		self.label = []
		with open(self.opt.label_name, 'r') as label_file:
			labels = label_file.readlines()
			for label in labels:
				self.label.append(label.strip())
		self.label[0] = 'item'

	def run(self):
		global detections, detection_lock, detection_stop

		cap = cv2.VideoCapture(opt.camera_id)
		if not cap.isOpened():
			print ('[Err] Camera device not found!')
			return

		window_name = 'Recognization and Grabbing System'
		cv2.namedWindow(window_name)
		# cv2.setMouseCallback(window_name, self.mouse_event)

		save_frame_name = '1.jpg'
		save_frame_path = os.path.join(self.opt.local_img_dir, save_frame_name)

		freeze_flag = False

		while cap.isOpened():
			# print ('test1')		
			status, frame = cap.read()

			cv2.imwrite(save_frame_path, frame)

			# if not freeze_flag:
			# 	if self.detectionQueue.empty():
			# 		detectionThread = DetectionThread(self.server, save_frame_path, self.opt, self.detectionQueue)
			# 		detectionThread.start()
			# 		detectionThread.join()
			# 	else:
			# 		self.detections = self.detectionQueue.get()

			if detections !=[] and detection_lock.acquire():
				
				# print (detections)
				try:
					count = 0
					for x1, y1, x2, y2, cls, conf in detections:
						if conf < self.opt.final_conf:
							continue
						if self.activate_id == count:
							plot_one_box([x1, y1, x2, y2], frame, label='%s:%.2f' % (self.label[int(cls)], conf), color=self.color_activate, line_thickness = 2)
						else:
							plot_one_box([x1, y1, x2, y2], frame, label='%s:%.2f' % (self.label[int(cls)], conf), color=self.color, line_thickness = 2)
						count += 1
				except:
					print ('None type detection')
				detection_lock.release()
						
			cv2.imshow(window_name, frame)
			key = cv2.waitKey(delay = 1)

			if key > 0 and chr(key) == 'c':
				freeze_flag = not freeze_flag 
			elif key > 0 and chr(key) == 'b':
				save_path = time.strftime('%Y_%m_%d_%H_%M_%S.jpg',time.localtime(time.time()))
				cv2.imwrite(save_path, frame)
			elif key > 0:
				print ('[Press any key except \'C\'] End detection!')
				return 0
		print ('test2')
		detection_stop = True
		cap.release()
		cv2.destroyAllWindows() 

	def mouse_event(self, event, x, y, flags, param):
		global detections, detection_lock
		detection_activate  = []
		activate_flag = False

		try:
			if detection_lock.acquire():
				count = 0
				for x1, y1, x2, y2, cls, conf in detections:
					if x > x1 and x < x2 and y > y1 and y < y2:
						detection_activate  = [x1, y1, x2, y2]
						self.activate_id = count
						activate_flag = True
						break
					count += 1
				detection_lock.release()
		except:
			return

		if not activate_flag:
			self.activate_id = -1

		if event == cv2.EVENT_LBUTTONDOWN:
			# send detection result to robot
			if detection_activate  != []:
				print ('Click to send!')
				grab(self.robotArm, detection_activate)
			return


if __name__ == '__main__':
	# Main Function
	parser = argparse.ArgumentParser()
	parser.add_argument('--server_addr', type=str, default='47.110.124.210', help='GPU server IP address')
	parser.add_argument('--private_key_file', type=str, default='detect\\hrstek_key.pem', help='path to private key file')
	parser.add_argument('--print_debug_info', type=bool, default=False, help='whether to print debug info or not')

	parser.add_argument('--local_img_dir', type=str, default='images', help='path to local client image directory')
	parser.add_argument('--local_result_dir', type=str, default='output', help='path to local client result directory')
	parser.add_argument('--target_server_dir', type=str, default='/root/project/server', help='path to server work directory')
	parser.add_argument('--target_detector_dir', type=str, default='/root/project/yolov3/', help='path to server detector directory')
	parser.add_argument('--log_dir', type=str, default='logs', help='path to log directory')

	parser.add_argument('--camera_id', type=int, default=0, help='camera device ID')
	parser.add_argument('--final_conf', type=float, default=0.8, help='final object confidence threshold')

	parser.add_argument('--label_name', type=str, default='detect\\coco.names', help='final object confidence threshold')

	opt = parser.parse_args()
	print(opt)

	for local_dir in [opt.local_img_dir, opt.local_result_dir, opt.log_dir]:
		if not os.path.exists(local_dir):
			os.mkdir(local_dir)

	# Writting log file
	log_file_name = time.strftime('%Y_%m_%d_%H_%M_%S.txt',time.localtime(time.time()))
	logfile = open(os.path.join(opt.log_dir, log_file_name), 'a')
	logfile.write('Logging INFO:\n')

	# Init Server
	private_key = paramiko.RSAKey.from_private_key_file(opt.private_key_file)
	server = Server(opt.server_addr, 22, 'root', private_key, print_debug_info = opt.print_debug_info)
	server.connect()

	# Init Robot Arm
	robotArm = ArmControl.RobotArm()
	robotArm.reset()

	# Show GUI
	guiThread = GUIThread(server, opt)
	detectionThread = DetectionThread(server, opt, robotArm)
	detectionThread.setDaemon(True)

	guiThread.start()
	detectionThread.start()

	guiThread.join()

	server.close()
	logfile.close()
