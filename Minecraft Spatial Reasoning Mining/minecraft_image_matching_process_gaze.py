## Overlaying circle onto Minecraft videos 

from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from matplotlib import pyplot as plt
import csv
import os
from scipy import stats

'''
parser = argparse.ArgumentParser(description='Code for Feature Detection tutorial.')
parser.add_argument('--input1', help='Path to input image 1.', default='box.png')
parser.add_argument('--input2', help='Path to input image 2.', default='box_in_scene.png')
args = parser.parse_args()

img1 = cv.imread(cv.samples.findFile(args.input1), cv.IMREAD_GRAYSCALE)
img2 = cv.imread(cv.samples.findFile(args.input2), cv.IMREAD_GRAYSCALE)
'''
#img1 = cv.imread('1_50 - Copy.jpg',0)          # queryImage
#folders = [a for a in os.listdir(".") if os.path.isdir(a)]
folders=['mc06']
#video_files = ['mc06\\01_screen.mp4','mc07\\01_compressed.mp4','posyx-famjam\\01_camera_screen_smaller.mp4', 'posyx-famjam\\02_camera_screen_smaller.mp4']
#gaze_files = ['mc06\\01_gazedata.stream~','mc07\\01_gazedata.stream~', 'posyx-famjam\\01_gazedata.stream~','posyx-famjam\\02_gazedata.stream~']
#video_files = ['mc01\\01_camera_screen.mp4', 'mc01\\02_camera_screen.mp4', 'mc02\\01_compressed.mp4','mc03\\01_compressed.mp4','mc03\\02_compressed.mp4', 'mc06\\01_screen','mc07\\01_compressed.mp4','posyz-famjam\\01_camera_screen_smaller.mp4', 'posyz-famjam\\02_camera_screen_smaller.mp4']
#gaze_files = ['mc01\\01_gazedata.stream~','mc01\\02_gazedata.stream~', 'mc02\\01_gazedata.stream~','mc03\\01_gazedata.stream~', 'mc03\\02_gazedata.stream~', 'mc06\\01_gazedata.stream~','mc07\\01_gazedata.stream~', 'posyz-famjam\\01_gazedata.stream~','posyz-famjam\\02_gazedata.stream~']
survey_file = "qualtrics_timing_data.csv"
survey_csv = open(survey_file)
csv_file =csv.DictReader(survey_csv)

for line in csv_file:
	parent_folder = line["parent_folder"]
	if parent_folder == '':
		continue
	folder_name = line["folder"]
	file_=line["file_name"]
	folder = os.path.join(parent_folder, folder_name)
	video_files = [f for f in os.listdir(folder) if ".mp4" in f]
	gaze_files = [f for f in os.listdir(folder) if "gazedata.stream~" in f]
	video_file = None
	gaze_file = None
	user_score = line["average_score"]
	user_incorrect = line["user_incorrect"]
	if file_ !="":
		video_file = video_files[0]
		gaze_file = gaze_files[0]
	else:
		video_file = [f for f in video_files if file_ in f][0]
		gaze_file =  [f for f in gaze_files if file_ in f][0]
	
	
	cap = cv.VideoCapture(os.path.join(folder,video_file))
	frame_counter = 0
	video_fps = 5.0
	if folder == 'mc06' in folder:
		print(folder)
		video_fps = 10.0
	images_files= []
	images = []
	image_keypoints  = []
	gazefile = open(os.path.join(folder, gaze_file), "r")
	list_of_datapoints = gazefile.readlines()
	#time_step = 0.2
	c_run_time = 0.0
	name = "_".join(["mmla",folder,video_file])
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
	
	print(frame_counter)