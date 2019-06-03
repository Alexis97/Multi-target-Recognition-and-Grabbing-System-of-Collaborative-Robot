#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Alexis Zhang

import argparse
import shutil
import time
import os
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *

import cv2

from PIL import Image

classes = ['items']
colors = [[255,0,0]]

def load_model(opt):
	# Initialize model and keep model running
	device = torch_utils.select_device()
	# Initialize model
	model = Darknet(opt.cfg, opt.img_size)

	# Load weights
	if opt.weights.endswith('.pt'):  # pytorch format
		model.load_state_dict(torch.load(opt.weights, map_location='cpu')['model'])
	else:  # darknet format
		_ = load_darknet_weights(model, opt.weights)
	
	# Get classes and colors
	# classes = load_classes(parse_data_cfg(data_cfg)['names'])
	# colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

	model.to(device).eval()

	return model

def detect(
		model,
		img_path,
		save_path, 
		img_size=416,
		conf_thres=0.3,
		nms_thres=0.45,
):
	device = torch_utils.select_device()
	# Read image
	img0 = cv2.imread(img_path)  # BGR
	assert img0 is not None, 'File Not Found ' + img_path

	# Padded resize
	img, _, _, _ = letterbox(img0, height=img_size)

	# Normalize RGB
	img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
	img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
	img /= 255.0  # 0 - 255 to 0.0 - 1.0

	

	t = time.time()
	print('image %s: ' % (img_path), end='')

	# Get detections
	img = torch.from_numpy(img).unsqueeze(0).to(device)
	pred = model(img)
	pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold

	if len(pred) > 0:
		# Run NMS on predictions
		detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]

		# Rescale boxes from 416 to true image size
		scale_coords(img_size, detections[:, :4], img0.shape).round()

		# Print results to screen
		unique_classes = detections[:, -1].cpu().unique()
		for c in unique_classes:
			n = (detections[:, -1].cpu() == c).sum()
			print('%g %ss' % (n, classes[int(c)]), end=', ')

		# Draw bounding boxes and labels of detections
		# Write to file
		with open(os.path.splitext(save_path)[0] + '.txt', 'w+') as file:
			for x1, y1, x2, y2, conf, cls_conf, cls in detections:
				file.write('%g %g %g %g %g %g\n' %
						   (x1, y1, x2, y2, cls, cls_conf * conf))

	dt = time.time() - t
	print('Done. (%.3fs)' % dt)


def tracking(model, opt):
	print ('Start tracking...')

	files_info = {}
	while True:
		# loop tracking
		
		update_list = []      # update list for new images

		files = os.listdir(opt.img_root)
		for file in files:
			flag_file = os.path.splitext(file)[0]
			if (file == flag_file): # IS a flag file
			# print ('Flag %s' % (file))
				continue
			try:                    # IS an updated image
				flag_abspath = os.path.join(opt.img_root, flag_file)
				flag = open(flag_abspath, 'r')
				flag.close()
				update_list.append(file)
			except:                 # NOT an updated image
				# print ('No update %s' % (file))
				continue

		if update_list:
			print (update_list)
			for img_path in update_list:
				img_abspath = os.path.join(opt.img_root, img_path)
				output_abspath = os.path.join(opt.output_root, img_path)
				flag_file = os.path.splitext(img_path)[0]
				flag_abspath = os.path.join(opt.img_root, flag_file)

				detect(model,
					img_abspath,
					output_abspath, 
					opt.img_size,
					opt.conf_thres,
					opt.nms_thres
					)
				os.remove(flag_abspath) 
		# else:
		#     print ('No update file...')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
	parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
	parser.add_argument('--img_root', type=str, default='/root/project/server/images', help='path to server image folder')
	parser.add_argument('--output_root', type=str, default='/root/project/server/output', help='path to server output folder')
	parser.add_argument('--data_cfg', type=str, default='cfg/coco.data', help='coco.data file path')
	parser.add_argument('--img_size', type=int, default=32 * 13, help='size of each image dimension')
	parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
	parser.add_argument('--nms_thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
	opt = parser.parse_args()
	print(opt)

	with torch.no_grad():
		yolo_model = load_model(opt)
		tracking(yolo_model, opt)


