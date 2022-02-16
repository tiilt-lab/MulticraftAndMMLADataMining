from __future__ import print_function
import cv2 as cv
import numpy as np
import csv
import os


survey_file = "data_output_excluding_questions.csv"
survey_csv = open(survey_file)
csv_file =csv.DictReader(survey_csv)

for line in csv_file:
	# ------------------------------------------------------------------------------------------------------------
	# Look at path from each line in csv file
	# Each line is a new path
	# ------------------------------------------------------------------------------------------------------------
	parent_folder = line["parent_folder"]
	if parent_folder == '':
		continue
	folder_name = line["folder"]

	# ------------------------------------------------------------------------------------------------------------
	# To filter based on what isn't denoted as "skip"
	# ------------------------------------------------------------------------------------------------------------
	if folder_name == "skip" or parent_folder == "skip":
		continue

	# ------------------------------------------------------------------------------------------------------------
	# Just a number because SSI labels the outputs as numbers, sometimes multiple files with
	# different numbers will be on there due to testing issues and SSI running multiple times
	# Each different number is a different time that SSI was ran
	# ------------------------------------------------------------------------------------------------------------
	file_=line["file_name"]
	folder = os.path.join(parent_folder, folder_name)

	# ------------------------------------------------------------------------------------------------------------
	# Get corresponding video and gaze file based on the path given in the csv and the number the file has
	# ------------------------------------------------------------------------------------------------------------
	video_files = [os.path.join(folder, f) for f in os.listdir(folder) if ".mp4" in f]
	gaze_files = [os.path.join(folder, f) for f in os.listdir(folder) if "gazedata.stream~" in f]
	video_file = None
	gaze_file = None

	# ------------------------------------------------------------------------------------------------------------
	# Get file
	# ------------------------------------------------------------------------------------------------------------
	if file_ == "":
		video_file = video_files[0]
		gaze_file = gaze_files[0]
	else:
		video_file = [f for f in video_files if file_ in f][0]
		gaze_file = [f for f in gaze_files if file_ in f][0]

	cap = cv.VideoCapture(video_file)
	frame_counter = 0
	video_fps = float(cap.get(cv.CAP_PROP_FPS)) #5.0
	num_gaze_per_frame = int(18/(video_fps/5.0))
	images_files= []
	images = []
	image_keypoints  = []
	gazefile = open(gaze_file, "r")
	list_of_datapoints = gazefile.readlines()
	#time_step = 0.2
	c_run_time = 0.0
	name = os.path.join(".\\processed_videos", video_file.replace("\\", "_"))
	fourcc =cv.VideoWriter_fourcc(*'mp4v')
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