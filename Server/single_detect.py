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

def detect(
        cfg,
        weights,
        img_path,
        output='output',  # output folder
        data_cfg = 'cfg/coco.data',
        img_size=416,
        conf_thres=0.3,
        nms_thres=0.45,
        save_txt=False,
        save_images=True
):
    device = torch_utils.select_device()
    if not os.path.exists(output):
        os.mkdir(output)
    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        if weights.endswith('yolov3.pt') and not os.path.exists(weights):
            if (platform == 'darwin') or (platform == 'linux'):
                os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    model.to(device).eval()

    # Read image
    img0 = cv2.imread(img_path)  # BGR
    assert img0 is not None, 'File Not Found ' + img_path

    # Padded resize
    img, _, _, _ = letterbox(img0, height=img_size)

    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    # colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]
    colors = [[0, 255, 0]]
    t = time.time()
    print('image %s: ' % (img_path), end='')
    save_path = str(os.path.join(output, os.path.split(img_path)[1]))

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
        for x1, y1, x2, y2, conf, cls_conf, cls in detections:
            if save_txt:  # Write to file
                with open(save_path + '.txt', 'a') as file:
                    file.write('%g %g %g %g %g %g\n' %
                               (x1, y1, x2, y2, cls, cls_conf * conf))

            # Add bbox to the image
            label = '%s %.2f' % (classes[int(cls)], conf)
            plot_one_box([x1, y1, x2, y2], img0, label=label, color=colors[int(cls)], line_thickness = 3)

    dt = time.time() - t
    print('Done. (%.3fs)' % dt)

    if save_images:  # Save generated image with detections
        cv2.imwrite(save_path, img0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
    parser.add_argument('--img_path', type=str, default='data/samples/1.png', help='path to image')
    parser.add_argument('--output', type=str, default='output', help='path to output folder')
    parser.add_argument('--data_cfg', type=str, default='cfg/coco.data', help='coco.data file path')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.weights,
            opt.img_path,
            output = opt.output,
            img_size=opt.img_size,
            data_cfg = opt.data_cfg,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres
        )

