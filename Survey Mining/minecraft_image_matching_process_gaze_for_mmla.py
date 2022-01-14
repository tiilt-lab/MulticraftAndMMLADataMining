from __future__ import print_function
import cv2 as cv
import numpy as np
import csv
import os


folders = [a for a in os.listdir(".") if os.path.isdir(a)]

for folder in folders:
	video_files = [f for f in os.listdir(folder) if ".mp4" in f]
	gaze_files = [f for f in os.listdir(folder) if "gazedata.stream~" in f]
	video_file = None
	gaze_file = None
	video_file = [f for f in video_files if file_ in f][0]
	gaze_file =  [f for f in gaze_files if file_ in f][0]


	cap = cv.VideoCapture(os.path.join(folder,video_file))
	frame_counter = 0
	video_fps = float(cap.get(cv.CAP_PROP_FPS)) #5.0
	num_gaze_per_frame = int(18/(video_fps/5.0))
	images_files= []
	images = []
	image_keypoints  = []
	gazefile = open(os.path.join(folder, gaze_file), "r")
	list_of_datapoints = gazefile.readlines()
	#time_step = 0.2
	c_run_time = 0.0
	name = "_".join(["mmla_1",folder,video_file])
	fourcc =cv.VideoWriter_fourcc(*'mpv4')
	out = cv.VideoWriter(name, fourcc, video_fps, (1280,720))
	while(cap.isOpened()):
		ret, frame=cap.read()
		matches=None
		if ret == True:
			start_index = int(90*frame_counter/video_fps)
			for idx in range(int(18/(video_fps/5))):
				if (start_index+idx) >= len(list_of_datapoints):
					break
				c_x, c_y, c_time=list_of_datapoints[start_index+idx].split()
				c_x=int(float(c_x))
				c_y=int(float(c_y))
				radius = 5
				center = (c_x-radius, c_y-radius)

				color = (0,255,0)
				thickness=2
				if c_x <= 2000 and c_y<=2000:
					ima = cv.circle(frame, center, radius, color, thickness)    
			out.write(frame)
			frame_counter+=1

		else:
			break