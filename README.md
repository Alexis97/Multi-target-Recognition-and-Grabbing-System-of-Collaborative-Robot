# Multi-target-Recognition-and-Grabbing-System-of-Collaborative-Robot #
## Introduction ##
* This project is based on the thesis [Multi-target Recognition and Grabbing System of Collaborative Robot Based on Machine Learning] by author Alexis (Shimian) Zhang.
* The project is running in separatedly three parts: a remote GPU server, a local PC client and a robotic platform.
* The code for remote server is under `$/Server` folder. 
  The code for local client is the main part of this project, which has a main function entrance in `$/demo.py`.
  The code for robotic platform could be various due to different implementation. In this project we use [HRSTEK](http://www.hrstek.com/) robot arm as platform.
  Figure shows the communication among these parts.
* For accomplishing recognition and grasp task, the whole system contains recognition subsystem and grasping subsystem.
  Figure shows the flow diagram of this project.
  Starting from a top-down viewed camera, it takes a photo of cluttered objects region in real time.
	The recognition system takes the image as input, and the well trained YOLO-v3 network runs to predict locations and categories of each object.
	The detection results are sent to a PC, and a GUI shows the bounding boxes and classes information for user to choose which one to pick.
	Grasping algorithm runs on PC converts the position and category information of candidate object into rotate parameters of robot arm joints.
	The command order is finally sent to robot arm to pick up the designated item. 
* In this project, we use YOLO-v3 as the backbone of recognition subsystem.
  We modify [ultralytics/yolov3](https://github.com/ultralytics/yolov3) to satisfy our multi-target real time object recognition.
  Figure shows the structure of YOLO-v3.
  We fine tune YOLO-v3 with PFN-PIC dataset to strengthen detection performance from a top-view camera of cluttered daily objects.
  PFN-PIC (PFN Picking Instructions for Commodities) is from Hatori, *et al.* in their work [Interactively Picking Real-World Objects with Unconstrained Spoken Language Instructions] which contains 1,180 images taken from top-view with bounding boxes and human instruction annotations. 
  Figure shows the detection result after fine tuning.
* For grasping subsystem, we apply coordinate transformation and inverse kinematics algorithm to achieve this task.
  Both the functions could be found in `$/grabbing/HrstekArmLibrary/ArmControl.py`.
  Figure shows a brief illustration of how coordinate transformation works.
  Inverse kinematics algorithm solves the problem of giving end position of a robot arm, to calculate the angles of each joints, which helps our grasping subsystem to send rotate command to the robotic platform.
  
## Usage ##
* This project could be used for industrial production of sorting various kinds of tiny components.
  And in security and protect, this implementation could be used to pick up a suspicious object and exclude it with remote operation by human beings.

## Demo ##
* Firstly, copy the `$/Server` folder to a GPU server (GPU memory > 4GB is recommanded). Run `$/Server/server_detect.py` on GPU server for continuously detecting images. The weight file for YOLO-v3 network is stored under `$/Server/weights`. Unfortunately, the fine tuned weight we use for this project is unaviliable on github now. Please refer to [Train] section below to fine tune your own network, or briefly download pretrained weights from [https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI].  
* Then Run `$/demo.py` on local client to start the demo of this project.

## Train ##
* To fine tune a YOLO-v3 network with PFN-PIC dataset, you need firstly download the pretrained weights from [https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI]. 
  Make sure the weight file is under *$/Server/weights* on GPU server. Then download PFN-PIC dataset from [https://github.com/
pfnet-research/picking-instruction], and convert it into COCO format for next training.
  Finally run `python3 train.py` under `$/Server` to start training.
  
## Future Work
* TODO: 
** Improve grasping algorithm.
** Apply instance segmentation for precisely grasping task.
** Apply object tracking technique for stability.
