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

#minHessian = 400
detector = cv.ORB_create(1000)
#images_roi_counter = {}

mapping_file = "qualtrics_mapping.csv" #file provides mapping between the question and the reference image
csv_file = open(mapping_file)
csv_data=csv.DictReader(csv_file)
images_files= [] #store the filename for each image
images = [] #stores all of the images

image_keypoints  = [] #variable to store keypoints for each reference image

'''
This chunk of code pulls in the images and then processes the keypoints and descriptors
for all of the reference images
'''
fixation_counter = {}
image_contours = {}
user_fixation_counter = {}
user_fixation_occ = {}
n_fixation_counter = {}
for line in csv_data:
	file_name=line['image_file']+"_"+line["angle"]
	if line['same']=="2":
		file_name+="_R.jpg"
	else:
		file_name+=".jpg"
	file_name = os.path.join("Allstimuliasjpg\\All stimuli as jpg_", file_name)
	images_files.append(file_name)
	#print(file_name)
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
	th3 = cv.adaptiveThreshold(img1,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,15,2)
	contours, hier = cv.findContours(th3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	#add all contouns to the fixation counter
	#within each Counter make a counter for above median, below median, for time and score
	image_contours[file_name]={}
	fixation_counter[file_name]={}
	for c in range(len(contours)):
		#print(contours[c])
		#print(hier[0][c])
		(x,y,w,h) = cv.boundingRect(contours[c])
		if (w * h) > 2000:
			#print(w*h)
			#c_contour = contours[c]
			fixation_counter[file_name][c]= Counter()
			image_contours[file_name][c] = contours[c]
			#cv.drawContours(c_img, contours, c, (0,255,0), 3)
	#cv.imshow('Matches', c_img )
	#print(frame_counter)
	#cv.waitKey()


''' This loop iterates over the video files and performs feature extraction on each frame of the video.
It also opens the corresponding gaze file and processes the data according to the associated video frame
'''
survey_file = "data_output_w_times_v3.csv"
survey_csv = open(survey_file)
csv_file =csv.DictReader(survey_csv)
all_indices = []
user_index = 0
user_data = {}
maxdist = 10
mindur = 5
user_fixations = {}
user_question_time = {}
image_times_list = {}
#add fixation revisits
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
	user_index = int(line['user'])
	user_score = float(line["average_score"]) <0
	user_incorrect = float(line["user_incorrect"]) <=1
	user_data[user_index]= (user_score, user_incorrect)
	user_fixation_counter[user_index]= Counter()
	n_fixation_counter[user_index]= Counter()
	user_fixation_occ[user_index]=Counter()
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
		g_t = i * 1000.0/90
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
	#print("processing video", video_file)
	frame_counter = 0
	user_gaze_points = {}
	question_start = int(line["first_question"])*video_fps
	c_question_duration = 0
	match_counter=0
	#print(video_file)
	fixation_sequence = []
	p_contours=[]
	n_fixation_counter_labels=[]
	p_fixation = []
	user_sequence = []

	while(cap.isOpened()):
		ret, frame=cap.read()
		#print(frame_counter, question_start, ret)

		if frame_counter>=question_start:
			#cv.imshow('Matches', frame)
			#print(frame_counter)
			#cv.waitKey()

			matches=None
			if ret == True and image_index<len(images_files):
				if image_index not in user_gaze_points:
					user_gaze_points[image_index]={}
				if frame_counter >= question_start:
					img2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
					img_copy = img2.copy()
					ret, thresh = cv.threshold(img2,127,255, cv.THRESH_BINARY_INV) #we use thresholding to find the bounding box
					contours, hier = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
					img_to_check = img2.copy()
					box_x = 0
					box_y =0
					box_w = 0
					box_h = 0
					cois = []
					contour_o_i = None
					coi_set = False
					for cnt in contours:
						(x,y,w,h) = cv.boundingRect(cnt)
						#print(x,y,w,h)
						if (w > 600 and w < 800) and ( h > 100 and h < 500):
							img_to_check = img2[y:y+h, x:x+w]  #this restricts the box over which to check from keypoints to the box that contains the reference image
							#print(img_to_check.shape, img1.shape)
							
							#print(dff, frame_counter)
						    # Apply template Matching
							#w_1, h_1 = img_to_check.shape[::-1]
							#img3=images[image_index].copy()
							#res = cv.matchTemplate(img3,img_to_check,cv.TM_SQDIFF)
							#min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
							# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
							#top_left = min_loc
							# Specify a threshold 
							#threshold = 0.1
							  
							# Store the coordinates of matched area in a numpy array 
							#loc = np.where( res < threshold)  
							#dff=cv.subtract(img_to_check, images[image_index][top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w])
							#print(cv.countNonZero(dff))
							# Draw a rectangle around the matched region. 
							#for pt in zip(*loc[::-1]): 
							#    cv.rectangle(img3, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2) 
							  
							# Show the final image with the matched area. 
							#cv.imshow('Detected',img3)
							#cv.waitKey()

							box_x = x
							box_y = y
							box_w = w
							contour_o_i = cnt
							coi_set = True
							box_h = h
							#print(w,w_1, h, h_1)
							#bottom_right = (top_left[0] + w, top_left[1] + h)

							#print(top_left, bottom_right, box_x, box_y)
							#raw_input()
							break
							#cv.rectangle(img_copy, (x,y),(x+w,y+h),(255, 0, 0),10)

					keypoints2, descriptors2 = detector.detectAndCompute(img_to_check, None)
					#this section looks at the level of similarity between the current frame and the previous frame. This helps us know when the stimulus has changed. 
					p_matches=0
					if type(p_dp)!=None.__class__ and p_dp.any():
						if type(descriptors2)!=None.__class__ and descriptors2.any():
							matches = matcher.match(p_dp, descriptors2)
							p_matches = len([match for match in matches if match.distance<35]) #The value of 35 was selected to account for some slight differences and seems to work well in practice
						#print("matches to previous",p_matches)
						else:
							p_matches= 0
					
					if type(descriptors2)!=None.__class__ and descriptors2.any():
						matches = matcher.match(image_keypoints[image_index]["descriptors"], descriptors2)
						keypoints1=image_keypoints[image_index]["keypoints"]
						img1 = images[image_index] # trainImage
						limited_matches = []	#not all of hte matches are accurate. limited matches
						keypoints_to_keep = []
						keypoints_offsets = {'x':[],'y':[]}
						# within this loop we keep track of the similarity between the reference image and the current stimulus image
						for match in matches:
							if match.distance<35: #35
								#check if keypoints are both on the same half (check x values) check if greater than box_x + w/2, or the image.shape[0]/2
								limited_matches.append(match)
								#print(frame_counter, match.trainIdx, match.queryIdx)
								#print(match.imgIdx)
								keypoints_offsets['x'].append(int(keypoints1[match.queryIdx].pt[0]) - keypoints2[match.trainIdx].pt[0]) #these lists keep track of distance between points of correspondance of the reference and frame
								keypoints_offsets['y'].append(int(keypoints1[match.queryIdx].pt[1]) - keypoints2[match.trainIdx].pt[1]) #these lists keep track of distance between points of correspondance of the reference and frame
						#print(np.std(keypoints_offsets['x']),np.std(keypoints_offsets['y']))
						x_median_offset = 0
						y_median_offset = 0
						#once we have the list of all corresponding points, we get the mode different, which practically corresponds to the most accurate mapping
						if len(stats.mode(np.array(keypoints_offsets['x'])).mode)>0:
							x_median_offset = stats.mode(np.array(keypoints_offsets['x'])).mode[0]
							y_median_offset = stats.mode(np.array(keypoints_offsets['y'])).mode[0]
						start_index = int(90*frame_counter/video_fps) #used to determine the start index for the gaze points that correspond to the current frame
						c_img =img1.copy()
						#this loop processes the gaze data for the current video frame. it transforms that data point to the corresponding point in the reference image.
						num_gaze_per_frame = 18
						if video_fps ==10.0:
							num_gaze_per_frame = 9
						c_fixation=[]
						for idx in range(num_gaze_per_frame):
							if (start_index+idx) >= len(list_of_datapoints):
								break
							c_x, c_y, c_time=list_of_datapoints[start_index+idx].split()
							p_x, p_y, P_time = list_of_datapoints[start_index+idx-1].split()
							c_x = int(float(c_x))
							c_y = int(float(c_y))
							p_x = int(float(p_x))
							p_y = int(float(p_y))
							p_x_os = int(p_x - box_x + x_median_offset)
							p_y_os = int(p_y - box_y + y_median_offset)
							c_x_os = int(c_x - box_x + x_median_offset)
							c_y_os = int(c_y - box_y + y_median_offset)

							#if (coi_set and cv.pointPolygonTest(contour_o_i,(c_x, c_y),False)>=0) or len(c_fixation)>0:
							#if (c_x > box_x) and (c_x < (box_x+box_w)) and (c_y > box_y) and (c_y < box_y+box_h):
							
							squared_distance = ((c_x-p_x)**2 + (c_y-p_y)**2)
							dist = 0.0
							if squared_distance > 0:
								dist = squared_distance**0.5
								#print(dist)
							#check 
							u_image = images_files[image_index]
							if dist <= maxdist:
								c_fixation.append((c_x,c_y))
								user_gaze_points[image_index][start_index+idx]=(c_x_os,c_y_os)
								

								if idx == num_gaze_per_frame-1:
									if p_fixation!=[]:
										x_sign = math.copysign(1.0, p_fixation[0]-(800/2.0)), math.copysign(1.0, c_x_os-(800/2.0))
										y_sign = math.copysign(1.0, p_fixation[1]-(427/2.0)), math.copysign(1.0, c_y_os-(427/2.0))
										if (image_index, "x",x_sign) not in all_indices:
											all_indices.append((image_index,"x", x_sign))
										if (image_index, "y", y_sign) not in all_indices:
											all_indices.append((image_index,"y", y_sign))
										if (image_index,"x", x_sign,"y", y_sign) not in all_indices:
											all_indices.append((image_index,"x", x_sign, "y", y_sign))
										user_fixation_counter[user_index][(image_index, "x",x_sign)]+=1
										user_fixation_counter[user_index][(image_index, "y", y_sign)]+=1
										user_fixation_counter[user_index][(image_index,"x", x_sign,"y", y_sign)]+=1
										
										user_sequence.append((image_index,"x", x_sign,"y", y_sign))
										#user_fixation_counter[user_index][(image_index,"dist",((c_x_os-p_fixation[0])**2 + (c_y_os-p_fixation[1])**2)**0.5)]
										#user_sequence.append(( math.copysign(1.0, c_x_os-(800/2.0)), math.copysign(1.0, c_y_os-(427/2.0))))
									p_fixation = [c_x_os, c_y_os]
									c_contours = []
									for cnt_index,cnt in image_contours[u_image].items():
										(x,y,w,h) = cv.boundingRect(cnt)
										imgc = img1.copy()
										#print(x,y,w,h, c_x_os, c_y_os, cv.pointPolygonTest(cnt,(c_x_os, c_y_os),0))
										if cv.pointPolygonTest(cnt,(c_x_os, c_y_os),1)>=-5:
											if (image_index, cnt_index) not in all_indices:
												all_indices.append((image_index, cnt_index))
											#print(image_index, cnt_index, len(c_fixation), p_x_os, p_y_os, x,y,w,h)
											#cv.drawContours(imgc, [cnt], 0, (0,255,0), 3)
											#cv.circle(imgc, (p_x_os-5, p_y_os-5), 5, (255,0,0), 2) 
											#cv.imshow("images", imgc) 
											#cv.waitKey()
											#print("added fixation)")
											user_fixation_counter[user_index][(image_index, cnt_index)]+=len(c_fixation)
											user_fixation_occ[user_index][(image_index, cnt_index)]+=1
											for p_contour_index in p_contours:
												n_fixation_counter[user_index][(image_index, p_contour_index, cnt_index)]+=1
												if (image_index, p_contour_index, cnt_index) not in n_fixation_counter_labels:
													n_fixation_counter_labels.append((image_index, p_contour_index, cnt_index))
											c_contours.append(cnt_index)

									p_contours=c_contours
							elif dist > maxdist:
								if len(c_fixation) >= mindur:							
									p_x_os = int(p_x - box_x + x_median_offset)
									p_y_os = int(p_y - box_y + y_median_offset)
									c_contours = []
									if p_fixation!=[]:
										x_sign = math.copysign(1.0, p_fixation[0]-(800/2.0)), math.copysign(1.0, c_x_os-(800/2.0))
										y_sign = math.copysign(1.0, p_fixation[1]-(427/2.0)), math.copysign(1.0, c_y_os-(427/2.0))
										if (image_index, "x",x_sign) not in all_indices:
											all_indices.append((image_index,"x", x_sign))
										if (image_index, "y", y_sign) not in all_indices:
											all_indices.append((image_index,"y", y_sign))
										if (image_index,"x", x_sign,"y", y_sign) not in all_indices:
											all_indices.append((image_index,"x", x_sign, "y", y_sign))
										user_fixation_counter[user_index][(image_index, "x",x_sign)]+=1
										user_fixation_counter[user_index][(image_index, "y", y_sign)]+=1
										user_fixation_counter[user_index][(image_index,"x", x_sign,"y", y_sign)]+=1
										#user_sequence.append(( math.copysign(1.0, c_x_os-(800/2.0)), math.copysign(1.0, c_y_os-(427/2.0))))
										user_sequence.append((image_index,"x", x_sign,"y", y_sign))

									p_fixation = [c_x_os, c_y_os]
									for cnt_index,cnt in image_contours[u_image].items():
										(x,y,w,h) = cv.boundingRect(cnt)
										imgc = img1.copy()
										#print(x,y,w,h, c_x_os, c_y_os, cv.pointPolygonTest(cnt,(c_x_os, c_y_os),0))
										if cv.pointPolygonTest(cnt,(p_x_os, p_y_os),1)>=-5:
											if (image_index, cnt_index) not in all_indices:
												all_indices.append((image_index, cnt_index))
											#print(image_index, cnt_index, len(c_fixation), p_x_os, p_y_os, x,y,w,h)
											#cv.drawContours(imgc, [cnt], 0, (0,255,0), 3)
											#cv.circle(imgc, (p_x_os-5, p_y_os-5), 5, (255,0,0), 2) 
											#cv.imshow("images", imgc) 
											#cv.waitKey()
											user_fixation_counter[user_index][(image_index, cnt_index)]+=len(c_fixation)
											user_fixation_occ[user_index][(image_index, cnt_index)]+=1
											for p_contour_index in p_contours:
												n_fixation_counter[user_index][(image_index, p_contour_index, cnt_index)]+=1
												if (image_index, p_contour_index, cnt_index) not in n_fixation_counter_labels:
													n_fixation_counter_labels.append((image_index, p_contour_index, cnt_index))
											c_contours.append(cnt_index)
											#fixation_sequence.append()
									p_contours = c_contours	

								c_fixation==[(c_x,c_y)]
							'''
							u_image = images_files[image_index]
							for cnt_index,cnt in image_contours[u_image].items():
								(x,y,w,h) = cv.boundingRect(cnt)
								imgc = img1.copy()
								if math.fabs(cv.pointPolygonTest(cnt,(c_x_os, c_y_os),1))<5:
									if (image_index, cnt_index) not in all_indices:
										all_indices.append((image_index, cnt_index))

									#print(image_index, cnt_index, len(c_fixation), c_x_os, c_y_os, x,y,w,h)
									cv.drawContours(imgc, [cnt], 0, (0,255,0), 3)
									cv.circle(imgc, (c_x_os-5, c_y_os-5), 5, (255,0,0), 2) 
									cv.imshow("images", imgc) 
									cv.waitKey()
									user_fixation_counter[user_index][(image_index, cnt_index)]+=1
							'''	
						#print(len(limited_matches))
						if len(limited_matches) >=5 or (user_index==9 and len(limited_matches)>=2): #other approach is to look how many clicks per page, and wait for those clicks to happen
							match_found=True
							#print("match found", match_counter)
							match_counter+=1
						#print(match_found, match_counter, p_matches)
						if (match_found ==True and p_matches <10 and match_counter>5) or (match_found==True and len(limited_matches)<=2 and p_matches<400): #4 for other tests - # or c_question_duration > video_fps * float(line["C"+str(image_index+1)]):
							#print(match_found ==True and p_matches <10 and match_counter>5, c_question_duration > video_fps * float(line["C"+str(image_index+1)]))
							c_question_duration+=1
							print(",".join([str(user_index),str(image_index)]+[str(s).replace("'","").replace(",",";") for s in user_sequence]))
							user_sequence=[]							
							match_found = False
							#image_times_list.append([str(user_index), str(image_index), str(frame_counter)])
							#if image_index ==0:
								#print(",".join([str(user_index), str(image_index-1), str(question_start)]))
							#print(",".join([str(user_index), str(image_index), str(frame_counter)]))
							image_index+=1
							match_counter=0
							p_contours=[]
	
					'''
					if image_index<len(images_files) and (user_index==9):
						img1 = images[image_index] # trainImage
						#img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
						img_matches = np.empty((max(img1.shape[0], img_to_check.shape[0]), img1.shape[1]+img_to_check.shape[1], 3), dtype=np.uint8)
						cv.drawMatches(img1, keypoints1, img_to_check, keypoints2, limited_matches, img_matches)
						#-- Show detected matches
						cv.imshow('Matches', img_matches)
						#print(frame_counter)
						cv.waitKey()
					'''
					p_kp,p_dp = keypoints2, descriptors2

			else:
				break
		frame_counter+=1
	#print(n_fixation_counter)
	#reaw_input()
	#csv_outfile = open(str(user_index)+"_gaze_output.csv","wb")
	#csv_out = csv.writer(csv_outfile, delimiter=",")
	#csv_out.writerows([[str(user_index), str(c_image_index), str(t),str(user_gaze_points[c_image_index][t][0]), str(user_gaze_points[c_image_index][t][1])] for c_image_index in user_gaze_points for t in user_gaze_points[c_image_index]])
	#csv_outfile.close()
	#user_index+=1
	#print("done processing video for", str(user_index))
#print(",".join(["user"]+[str(a).replace(",",";") for a in all_indices] + ["occ_"+str(a).replace(",",";") for a in all_indices] + ["avg_"+str(a).replace(",",";") for a in all_indices] + [str(a).replace(",",";") for a in n_fixation_counter_labels] + ["score","wrong"]))
csv_outfile = open("fixation_contour_occ_output_inline_v6.csv","wb")
csv_out = csv.writer(csv_outfile, delimiter=",")
csv_out.writerow(["user"]+[str(a).replace(",",";") for a in all_indices] + ["occ_"+str(a).replace(",",";") for a in all_indices if "x" not in a and "y" not in a] + ["avg_"+str(a).replace(",",";") for a in all_indices if "x" not in a and "y" not in a] + [str(a).replace(",",";") for a in n_fixation_counter_labels] + ["score","wrong"])
#for user in user_fixation_occ:
#	print(",".join([str(user)]+[str(user_fixation_counter[user][a]) for a in all_indices] + [str(user_fixation_occ[user][a]) for a in all_indices]+ [str(user_fixation_counter[user][a]/max(user_fixation_occ[user][a],1.0)) for a in all_indices] + [str(n_fixation_counter[user][a]) for a in n_fixation_counter_labels] + [str(user_data[user][0]),str(user_data[user][1])]))
csv_out.writerows([[str(user)]+[str(user_fixation_counter[user][a]) for a in all_indices] + [str(user_fixation_occ[user][a]) for a in all_indices if "x" not in a and "y" not in a]+ [str(user_fixation_counter[user][a]/max(user_fixation_occ[user][a],1.0)) for a in all_indices if "x" not in a and "y" not in a] + [str(n_fixation_counter[user][a]) for a in n_fixation_counter_labels] + [str(user_data[user][0]),str(user_data[user][1])] for user in user_fixation_occ])
csv_outfile.close()

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