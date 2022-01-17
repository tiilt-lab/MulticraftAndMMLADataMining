## Just seems to be finding matches, not sure if there is any particular thing being taken from this

from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from matplotlib import pyplot as plt
import csv
import os
from scipy import stats
from collections import Counter
import math

def check_bounding_box(box_x,box_y,box_w,box_h,c_x, c_y):
	if (c_x > box_x) and (c_x < (box_x+box_w)) and (c_y > box_y) and (c_y < box_y+box_h):
		return True
	return False

def blink_detection(x, y, time, missing=0.0, minlen=10):
	
	"""Detects blinks, defined as a period of missing data that lasts for at
	least a minimal amount of samples
	
	arguments

	x		-	np array of x positions
	y		-	np array of y positions
	time		-	np array of EyeTribe timestamps

	keyword arguments

	missing	-	value to be used for missing data (default = 0.0)
	minlen	-	integer indicating the minimal amount of consecutive
				missing samples
	
	returns
	Sblk, Eblk
				Sblk	-	list of lists, each containing [starttime]
				Eblk	-	list of lists, each containing [starttime, endtime, duration]
	"""
	
	# empty list to contain data
	Sblk = []
	Eblk = []
	
	# check where the missing samples are
	mx = np.array(x==missing, dtype=int)
	my = np.array(y==missing, dtype=int)
	miss = np.array((mx+my) == 2, dtype=int)
	
	# check where the starts and ends are (+1 to counteract shift to left)
	diff = np.diff(miss)
	starts = np.where(diff==1)[0] + 1
	ends = np.where(diff==-1)[0] + 1
	
	# compile blink starts and ends
	for i in range(len(starts)):
		# get starting index
		s = starts[i]
		# get ending index
		if i < len(ends):
			e = ends[i]
		elif len(ends) > 0:
			e = ends[-1]
		else:
			e = -1
		# append only if the duration in samples is equal to or greater than
		# the minimal duration
		if e-s >= minlen:
			# add starting time
			Sblk.append([time[s]])
			# add ending time
			Eblk.append([time[s],time[e],time[e]-time[s]])
	
	return Sblk, Eblk

def remove_missing(x, y, time, missing):
	mx = np.array(x==missing, dtype=int)
	my = np.array(y==missing, dtype=int)
	x = x[(mx+my) != 2]
	y = y[(mx+my) != 2]
	time = time[(mx+my) != 2]
	return x, y, time


def fixation_detection(x, y, time, missing=0.0, maxdist=25, mindur=50):
	
	"""Detects fixations, defined as consecutive samples with an inter-sample
	distance of less than a set amount of pixels (disregarding missing data)
	
	arguments

	x		-	np array of x positions
	y		-	np array of y positions
	time		-	np array of EyeTribe timestamps

	keyword arguments

	missing	-	value to be used for missing data (default = 0.0)
	maxdist	-	maximal inter sample distance in pixels (default = 25)
	mindur	-	minimal duration of a fixation in milliseconds; detected
				fixation cadidates will be disregarded if they are below
				this duration (default = 100)
	
	returns
	Sfix, Efix
				Sfix	-	list of lists, each containing [starttime]
				Efix	-	list of lists, each containing [starttime, endtime, duration, endx, endy]
	"""

	x, y, time = remove_missing(x, y, time, missing)

	# empty list to contain data
	Sfix = []
	Efix = []
	
	# loop through all coordinates
	si = 0
	fixstart = False
	for i in range(1,len(x)):
		# calculate Euclidean distance from the current fixation coordinate
		# to the next coordinate
		squared_distance = ((x[si]-x[i])**2 + (y[si]-y[i])**2)
		dist = 0.0
		if squared_distance > 0:
			dist = squared_distance**0.5
		# check if the next coordinate is below maximal distance
		if dist <= maxdist and not fixstart:
			# start a new fixation
			si = 0 + i
			fixstart = True
			Sfix.append([time[i]])
		elif dist > maxdist and fixstart:
			# end the current fixation
			fixstart = False
			# only store the fixation if the duration is ok
			if time[i-1]-Sfix[-1][0] >= mindur:
				Efix.append([Sfix[-1][0], time[i-1], time[i-1]-Sfix[-1][0], x[si], y[si]])
			# delete the last fixation start if it was too short
			else:
				Sfix.pop(-1)
			si = 0 + i
		elif not fixstart:
			si += 1
	#add last fixation end (we can lose it if dist > maxdist is false for the last point)
	if len(Sfix) > len(Efix):
		Efix.append([Sfix[-1][0], time[len(x)-1], time[len(x)-1]-Sfix[-1][0], x[si], y[si]])
	return Sfix, Efix


def saccade_detection(x, y, time, missing=0.0, minlen=5, maxvel=40, maxacc=340):
	
	"""Detects saccades, defined as consecutive samples with an inter-sample
	velocity of over a velocity threshold or an acceleration threshold
	
	arguments

	x		-	np array of x positions
	y		-	np array of y positions
	time		-	np array of tracker timestamps in milliseconds

	keyword arguments

	missing	-	value to be used for missing data (default = 0.0)
	minlen	-	minimal length of saccades in milliseconds; all detected
				saccades with len(sac) < minlen will be ignored
				(default = 5)
	maxvel	-	velocity threshold in pixels/second (default = 40)
	maxacc	-	acceleration threshold in pixels / second**2
				(default = 340)
	
	returns
	Ssac, Esac
			Ssac	-	list of lists, each containing [starttime]
			Esac	-	list of lists, each containing [starttime, endtime, duration, startx, starty, endx, endy]
	"""
	x, y, time = remove_missing(x, y, time, missing)

	# CONTAINERS
	Ssac = []
	Esac = []

	# INTER-SAMPLE MEASURES
	# the distance between samples is the square root of the sum
	# of the squared horizontal and vertical interdistances
	intdist = (np.diff(x)**2 + np.diff(y)**2)**0.5
	# get inter-sample times
	inttime = np.diff(time)
	# recalculate inter-sample times to seconds
	inttime = inttime / 1000.0
	
	# VELOCITY AND ACCELERATION
	# the velocity between samples is the inter-sample distance
	# divided by the inter-sample time
	vel = intdist / inttime
	# the acceleration is the sample-to-sample difference in
	# eye movement velocity
	acc = np.diff(vel)

	# SACCADE START AND END
	t0i = 0
	stop = False
	while not stop:
		# saccade start (t1) is when the velocity or acceleration
		# surpass threshold, saccade end (t2) is when both return
		# under threshold
	
		# detect saccade starts
		sacstarts = np.where((vel[1+t0i:] > maxvel).astype(int) + (acc[t0i:] > maxacc).astype(int) >= 1)[0]
		if len(sacstarts) > 0:
			# timestamp for starting position
			t1i = t0i + sacstarts[0] + 1
			if t1i >= len(time)-1:
				t1i = len(time)-2
			t1 = time[t1i]
			
			# add to saccade starts
			Ssac.append([t1])
			
			# detect saccade endings
			sacends = np.where((vel[1+t1i:] < maxvel).astype(int) + (acc[t1i:] < maxacc).astype(int) == 2)[0]
			if len(sacends) > 0:
				# timestamp for ending position
				t2i = sacends[0] + 1 + t1i + 2
				if t2i >= len(time):
					t2i = len(time)-1
				t2 = time[t2i]
				dur = t2 - t1

				# ignore saccades that did not last long enough
				if dur >= minlen:
					# add to saccade ends
					Esac.append([t1, t2, dur, x[t1i], y[t1i], x[t2i], y[t2i]])
				else:
					# remove last saccade start on too low duration
					Ssac.pop(-1)

				# update t0i
				t0i = 0 + t2i
			else:
				stop = True
		else:
			stop = True
	
	return Ssac, Esac


#minHessian = 400
detector = cv.ORB_create(1000)
#images_roi_counter = {}

mapping_file = "qualtrics_mapping.csv" #file provides mapping between the question and the reference image
csv_file = open(mapping_file)
csv_data=csv.DictReader(csv_file)
images_files= ["portfolio_image.png"] #store the filename for each image
images = [] #stores all of the images

image_keypoints  = [] #variable to store keypoints for each reference image

'''
This chunk of code pulls in the images and then processes the keypoints and descriptors
for all of the reference images
'''
fixation_counter = {}
image_contours = {}
user_fixation_counter = {}
for line in csv_data:
	file_name = "portfolio_image.png"
	#images_files.append(file_name)
	#images_roi_counter[file_name]=Counter()
	c_img = cv.imread(file_name,1)
	img1 = c_img.copy()
	img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
 # trainImage
	#img_mat = np.empty((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)

	#cv.imshow(img1, img_mat)
	images.append(img1)

	kp,dp = detector.detectAndCompute(img1, None)
	image_keypoints.append({"keypoints": kp,"descriptors": dp})
	#ret, thresh = cv.threshold(img1,127,255, cv.THRESH_BINARY) #we use thresholding to find the bounding box
	


''' This loop iterates over the video files and performs feature extraction on each frame of the video.
It also opens the corresponding gaze file and processes the data according to the associated video frame
'''
survey_file = "qualtrics_timing_data.csv"
survey_csv = open(survey_file)
csv_file =csv.DictReader(survey_csv)
all_indices = []
user_index = 0
user_data = []
maxdist = 10
mindur = 5
for line in csv_file:

	parent_folder = line["parent_folder"]
	if parent_folder == '':
		continue
	folder_name = line["folder"]
	file_=line["file_name"]
	folder = os.path.join(parent_folder, folder_name)
	video_files = [os.path.join(folder,f) for f in os.listdir(folder) if ".mp4" in f]
	gaze_files = [os.path.join(folder,f) for f in os.listdir(folder) if "gazedata.stream~" in f]
	video_file = None
	gaze_file = None
	user_score = float(line["average_score"]) <0
	user_incorrect = float(line["user_incorrect"]) <=1
	user_data.append((user_score, user_incorrect))
	user_fixation_counter[user_index]= Counter()
	if file_ =="":
		video_file = video_files[0]
		gaze_file = gaze_files[0]
	else:
		video_file = [f for f in video_files if file_ in f][0]
		gaze_file =  [f for f in gaze_files if file_ in f][0]
	cap = cv.VideoCapture(video_file)
	video_fps = float(cap.get(cv.CAP_PROP_FPS)) #5.0
	gazefile = open(gaze_file, "r")
	list_of_datapoints = gazefile.readlines()
	gaze_x = []
	gaze_y = []
	gaze_time = []
	'''
	for i in range(len(list_of_datapoints)):
		g_x,g_y,t = list_of_datapoints[i].split()
		if x > 2000:
			g_x = 0.0
		if y > 2000:
			g_y = 0.0
		g_t = i * 1.0/90
		gaze_x.append(g_x)
		gaze_y.append(g_y)
		gaze_time.append(g_t)
	sfix, efix = fixation_detection(np.array(gaze_x), np.array(gaze_y), np.array(gaze_time), missing=0.0, maxdist=25, mindur=50)
	ssacc, fsacc = saccade_detection(gaze_x, gaze_y, gaze_time, missing=0.0, minlen=5, maxvel=40, maxacc=340)
	#time_step = 0.2
	'''
	c_run_time = 0.0

	match_found = False
	image_index=0
	p_kp, p_dp = np.array([]),np.array([])
	matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
	''' the loop below pull the frame from each video and then find the rectangle that contains the stimulus
	This is done by finding the contours, and looking for a contour of a specific size. Once that is found the x,y, height 
	and width to use as an offset later on
	'''
	print("processing video", video_file)

	frame_counter = 0
	while(cap.isOpened()):
		ret, frame=cap.read()
		matches=None
		if ret == True: # and image_index<len(images_files):
			img2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			#print(frame.shape)
			img_copy = img2.copy()
			ret, thresh = cv.threshold(img2,127,255, cv.THRESH_BINARY_INV) #we use thresholding to find the bounding box
			img_to_check = img2.copy()
			keypoints2, descriptors2 = detector.detectAndCompute(img_to_check, None)
			if type(descriptors2)!=None.__class__ and descriptors2.any():
				matches = matcher.match(image_keypoints[image_index]["descriptors"], descriptors2)
			#this section looks at the level of similarity between the current frame and the previous frame. This helps us know when the stimulus has changed. 
			keypoints1=image_keypoints[image_index]["keypoints"]
			img1 = images[image_index] # trainImage
			limited_matches = []	#not all of hte matches are accurate. limited matches
			# within this loop we keep track of the similarity between the reference image and the current stimulus image
			for match in matches:
				print(match)
				if match.distance<35: #35
					limited_matches.append(match)
					#print(frame_counter, match.trainIdx, match.queryIdx)
					#print(match.imgIdx)
			p_kp,p_dp = keypoints2, descriptors2
			frame_counter+=1
		else:
			break

	user_index+=1
	#print("done processing video for", str(user_index))
print(",".join(["user"]+[str(a).replace(",",";") for a in all_indices] + ["score","wrong"]))
for user in user_fixation_counter:
	print(",".join([str(user)]+[str(user_fixation_counter[user][a]) for a in all_indices] + [str(user_data[user][0]),str(user_data[user][1])]))
'''
for img_file in images_files:
	c_image = cv.imread(img_file,1)
	cv.cvtColor(c_image, cv.COLOR_BGR2HSV)
	#thickness = 2
	for cnt_index,cnt in image_contours[u_image].items():
		most_common = fixation_counter[img_file].most_common(1000)
	#print(most_common)
	for k in most_common:
		radius = 3
		thickness = 2
		color = (179, int(k[1]),int(k[1]))
		cv.circle(c_image,(k[0][0], k[0][1]),radius, color, thickness)
	#cv.imshow("image_output", c_img)
'''