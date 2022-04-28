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
from detectors import *
import time
import json

# TODO: Add quadrant addition display logic and look at it for the adjusted code

#minHessian = 400
detector = cv.ORB_create(1000)
#images_roi_counter = {}

# mapping_file: file provides mapping between the question and the reference image
def get_images_and_keypoints(mapping_file): 
    image_files = [] 
    # image_files = sorted([os.path.join("portfolio_stimuli", f) for f in os.listdir("portfolio_stimuli") if os.path.isfile(os.path.join("portfolio_stimuli", f))]) #store the filename for each image
    images = [] #stores all of the images 

    image_keypoints  = [] #variable to store keypoints for each reference image
    csv_file = open(mapping_file)
    csv_data=csv.DictReader(csv_file)
    for line in csv_data:
        file_name=line['image_file']+"_"+line["angle"]
        if line['same']=="2":
            file_name+="_R.jpg"
        else:
            file_name+=".jpg"
        file_name = os.path.join("..\\Allstimuliasjpg\\All stimuli as jpg_", file_name)
        image_files.append(file_name)

        c_img = cv.imread(file_name, 1)
        img1 = c_img.copy()
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        # trainImage
        #img_mat = np.empty((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)

        #cv.imshow(img1, img_mat)
        images.append(img1)
        kp, dp = detector.detectAndCompute(img1, None)
        image_keypoints.append({"keypoints": kp, "descriptors": dp})
    return image_files, images, image_keypoints


def get_files(line): 
    # ------------------------------------------------------------------------------------------------------------
    # Look at path from each line in csv file 
    # Each line is a new path 
    # ------------------------------------------------------------------------------------------------------------
    parent_folder = line["parent_folder"]
    if parent_folder == '':
        return None, None
    folder_name = line["folder"]

    # ------------------------------------------------------------------------------------------------------------
    # To filter based on what isn't denoted as "skip"
    # ------------------------------------------------------------------------------------------------------------
    if folder_name == "skip" or parent_folder == "skip": 
        return None, None

    # TODO: change the testing code
    # if not ((folder_name == "mc01" or folder_name == "mc06") and parent_folder == "other"):
    #     return None, None

    # ------------------------------------------------------------------------------------------------------------
    # Just a number because SSI labels the outputs as numbers, sometimes multiple files with 
    # different numbers will be on there due to testing issues and SSI running multiple times 
    # Each different number is a different time that SSI was ran 
    # ------------------------------------------------------------------------------------------------------------
    file_=line["file_name"]
    folder = os.path.join("..\\videos", os.path.join(parent_folder, folder_name))

    # ------------------------------------------------------------------------------------------------------------
    # Get corresponding video and gaze file based on the path given in the csv and the number the file has
    # ------------------------------------------------------------------------------------------------------------
    video_files = [os.path.join(folder,f) for f in os.listdir(folder) if ".mp4" in f]
    gaze_files = [os.path.join(folder,f) for f in os.listdir(folder) if "gazedata.stream~" in f]
    video_file = None
    gaze_file = None

    # ------------------------------------------------------------------------------------------------------------
    # Get file
    # ------------------------------------------------------------------------------------------------------------
    if file_ =="":
        video_file = video_files[0]
        gaze_file = gaze_files[0]
    else:
        video_file = [f for f in video_files if file_ in f][0]
        gaze_file =  [f for f in gaze_files if file_ in f][0]
    return video_file, gaze_file

def process_gaze_datapoints(gaze_file, video_fps): 
    gazefile = open(gaze_file, "r")
    list_of_datapoints = gazefile.readlines()

    # ------------------------------------------------------------------------------------------------------------
    # Get datapoint coordinates and find fixations and saccades 
    # ------------------------------------------------------------------------------------------------------------
    gaze_x = []
    gaze_y = []
    gaze_time = []
    for i in range(len(list_of_datapoints)):
        points = list_of_datapoints[i].split()
        if len(points) != 3: 
            continue
        g_x,g_y,t = points
        if float(g_x) > 1280:
            g_x = 0.0
        if float(g_y) > 800:
            g_y = 0.0
        g_t = i * 1000.0/90
        gaze_x.append(float(g_x))
        gaze_y.append(float(g_y))
        gaze_time.append(g_t)

    data_ts = filter_gaze(np.array(gaze_x), np.array(gaze_y), np.array(gaze_time), video_fps)
    return data_ts

def get_point_and_time(fsacc, c_fixation_index, question_start, video_fps): 
    c_fix_start_time, c_fix_end_time, c_x, c_y = fsacc[c_fixation_index]

    # ------------------------------------------------------------------------------------------------------------
    # Skipping over data until question starts 
    # ------------------------------------------------------------------------------------------------------------
    while (int(c_fix_end_time) < question_start):
        c_fixation_index += 1
        c_fix_start_time, c_fix_end_time, c_x, c_y = fsacc[c_fixation_index]

    og_start_time = c_fix_start_time
    og_end_time = c_fix_end_time
    lc_x = []
    lc_y = []
    while len(fsacc) > c_fixation_index:
        c_fix_start_time, c_fix_end_time, c_x, c_y = fsacc[c_fixation_index]
        if not og_start_time == c_fix_start_time:
            break
        lc_x.append(c_x)
        lc_y.append(c_y)
        c_fixation_index += 1

    return og_start_time, og_end_time, lc_x, lc_y, c_fixation_index
    

def get_box_of_interest(img2, contours): 
    box_x = 0
    box_y = 0
    img_to_check = img2.copy()
    ret_cnt = None

    # ------------------------------------------------------------------------------------------------------------
    # Look at each contour
    # ------------------------------------------------------------------------------------------------------------
    for cnt in contours:
        (x,y,w,h) = cv.boundingRect(cnt)
        # print(x,y,w,h)

        # ------------------------------------------------------------------------------------------------------------
        # Check if contour is in the box
        # ------------------------------------------------------------------------------------------------------------
        # TODO: 500 was good for portfolio
        if (w > 600 and w < 800) and (h > 100 and h < 500):
            img_to_check = img2[y:y+h, x:x+w]  #this restricts the box over which to check from keypoints to the box that contains the reference image
            # with_dot = img2[y:y+h, x:x+w]

            box_x = x
            box_y = y
            ret_cnt = [cnt]
            break
            #cv.rectangle(img_copy, (x,y),(x+w,y+h),(255, 0, 0),10)
    return box_x, box_y, img_to_check, ret_cnt


def get_pmatches(matcher, p_dp, descriptors2, match_thresh):
    #this section looks at the level of similarity between the current frame and the previous frame. This helps us know when the stimulus has changed. 
    p_matches=0
    if type(p_dp)!=None.__class__ and p_dp.any():
        if type(descriptors2)!=None.__class__ and descriptors2.any():
            matches = matcher.match(p_dp, descriptors2)
            # TODO: Maybe increase to 50
            p_matches = len([match for match in matches if match.distance<match_thresh]) #The value of 35 was selected to account for some slight differences and seems to work well in practice
        #print("matches to previous",p_matches)
        else:
            p_matches= 0
    return p_matches

# image_descriptor: image_keypoints[image_index]["descriptor"]
def get_limited_matches(matcher, image_descriptor, descriptors2, dist_thresh):
    matches = matcher.match(image_descriptor, descriptors2)
    limited_matches = []	#not all of the matches are accurate. limited matches
    # keypoints1 = image_keypoint["keypoints"]
    # within this loop we keep track of the similarity between the reference image and the current stimulus image
    
    # ------------------------------------------------------------------------------------------------------------
    # Get offsets to create mapping from gaze data in video to the reference image from all_stimuli 
    # We are doing this because the analysis is done on the all_stimuli image rather than the current frame 
    # ------------------------------------------------------------------------------------------------------------
    for match in matches:
        # TODO: Maybe increase to 50
        if match.distance<dist_thresh: #35
            #check if keypoints are both on the same half (check x values) check if greater than box_x + w/2, or the image.shape[0]/2
            limited_matches.append(match)
            
    return limited_matches


def get_median_offset(matcher, image_keypoint, descriptors2, keypoints2, dist_thresh):
    matches = matcher.match(image_keypoint["descriptors"], descriptors2)
    keypoints1 = image_keypoint["keypoints"]
    keypoints_offsets = {'x':[],'y':[]}

    for match in matches:
        # TODO: Maybe increase to 50
        if match.distance<dist_thresh: #35
            keypoints_offsets['x'].append(int(keypoints1[match.queryIdx].pt[0]) - keypoints2[match.trainIdx].pt[0]) #these lists keep track of distance between points of correspondance of the reference and frame
            keypoints_offsets['y'].append(int(keypoints1[match.queryIdx].pt[1]) - keypoints2[match.trainIdx].pt[1]) #these lists keep track of distance between points of correspondance of the reference and frame
    x_median_offset = 0
    y_median_offset = 0
    #once we have the list of all corresponding points, we get the mode different, which practically corresponds to the most accurate mapping
    if len(stats.mode(np.array(keypoints_offsets['x'])).mode)>0:
        x_median_offset = stats.mode(np.array(keypoints_offsets['x'])).mode[0]
        y_median_offset = stats.mode(np.array(keypoints_offsets['y'])).mode[0]
    return x_median_offset, y_median_offset


def get_c_coords(c_x, c_y, x_offset, y_offset): 
    c_x = int(float(c_x))
    c_y = int(float(c_y))
    c_x_os = int(c_x + x_offset)
    c_y_os = int(c_y + y_offset)
    return c_x_os, c_y_os


def find_quadrants(ret, c_img, c_x_os, c_y_os):
    x = 0
    y = 0
    (h, w) = c_img.shape

    one = np.array([[x, y], [x, h // 2], [w // 2, h // 2], [w // 2, y]]).reshape((-1, 1, 2)).astype(np.int32)
    two = np.array([[x + w // 2, y], [x + w // 2, h // 2], [x + w, h // 2], [x + w, y]]).reshape((-1, 1, 2)).astype(
        np.int32)
    three = np.array([[x, y + h // 2], [x, y + h], [x + w // 2, h], [x + w // 2, y + h // 2]]).reshape((-1, 1, 2)).astype(
        np.int32)
    four = np.array([[x + w // 2, y + h // 2], [x + w // 2, y + h], [w, h], [x + w, y + h // 2]]).reshape(
        (-1, 1, 2)).astype(np.int32)

    # sh = cv.circle(c_img, (c_x_os, c_y_os), 20, (255, 0, 0))
    # sh = cv.rectangle(sh, (x, y), (w//2, h//2), (255, 0, 0))
    # cv.imshow("sh", sh)
    # cv.waitKey(0)

    if cv.pointPolygonTest(one, (c_x_os, c_y_os), 1) >= -5:
        ret.append(1)
    elif cv.pointPolygonTest(two, (c_x_os, c_y_os), 1) >= -5:
        ret.append(2)
    elif cv.pointPolygonTest(three, (c_x_os, c_y_os), 1) >= -5:
        ret.append(3)
    elif cv.pointPolygonTest(four, (c_x_os, c_y_os), 1) >= -5:
        ret.append(4)

    # Breakpoint here and start testing to see if it works
    # print(ret)

    return ret

# c_img: images[image_index]
def populate_quadrants(quadrants, fk_q, image_index, quad_list):
    quad_list = list(set(quad_list))
    if image_index not in quadrants[fk_q]: 
        quadrants[fk_q][image_index] = []
    
    # sh = cv.circle(c_img, (c_x_os, c_y_os), 20, (255, 0, 0)) 
    # sh = cv.rectangle(sh, (x, y), (w//2, h//2), (255, 0, 0))
    # cv.imshow("sh", sh)
    # cv.waitKey(0)

    for q_ind in quad_list:
        quadrants[fk_q][image_index].append(q_ind)
    
    return quadrants


# # c_img: images[image_index]
# def populate_quadrants(quadrants, c_img, c_x_os, c_y_os, fk_q, image_index):
#     x = 0
#     y = 0
#     (h, w) = c_img.shape
#
#     one = np.array([[x, y], [x, h // 2], [w // 2, h // 2], [w // 2, y]]).reshape((-1, 1, 2)).astype(np.int32)
#     two = np.array([[x + w // 2, y], [x + w // 2, h // 2], [x + w, h // 2], [x + w, y]]).reshape((-1, 1, 2)).astype(
#         np.int32)
#     three = np.array([[x, y + h // 2], [x, y + h], [x + w // 2, h], [x + w // 2, y + h // 2]]).reshape(
#         (-1, 1, 2)).astype(np.int32)
#     four = np.array([[x + w // 2, y + h // 2], [x + w // 2, y + h], [w, h], [x + w, y + h // 2]]).reshape(
#         (-1, 1, 2)).astype(np.int32)
#
#     if image_index not in quadrants[fk_q]:
#         quadrants[fk_q][image_index] = []
#
#     # sh = cv.circle(c_img, (c_x_os, c_y_os), 20, (255, 0, 0))
#     # sh = cv.rectangle(sh, (x, y), (w//2, h//2), (255, 0, 0))
#     # cv.imshow("sh", sh)
#     # cv.waitKey(0)
#
#     if cv.pointPolygonTest(one, (c_x_os, c_y_os), 1) >= -5:
#         quadrants[fk_q][image_index].append(1)
#     elif cv.pointPolygonTest(two, (c_x_os, c_y_os), 1) >= -5:
#         quadrants[fk_q][image_index].append(2)
#     elif cv.pointPolygonTest(three, (c_x_os, c_y_os), 1) >= -5:
#         quadrants[fk_q][image_index].append(3)
#     elif cv.pointPolygonTest(four, (c_x_os, c_y_os), 1) >= -5:
#         quadrants[fk_q][image_index].append(4)
#
#     return quadrants

def check_match(limited_matches, match_counter, match_found, l_matches):
    # ------------------------------------------------------------------------------------------------------------
    # Check if at least 5 matches because not all are accurate
    # ------------------------------------------------------------------------------------------------------------
    # TODO: Might need to decrease to 3 
    if len(limited_matches) >= l_matches: #other approach is to look how many clicks per page, and wait for those clicks to happen
        match_found=True
        #print("match found", match_counter)
        match_counter+=1
    return match_found, match_counter

def check_no_match(limited_matches, p_matches, match_counter, match_found, image_index): 
    # ------------------------------------------------------------------------------------------------------------
    # match_found is a mode. If the mode is switched on and these other conditions apply, then that means there is 
    # no match found anymore and the match_found should be false. 
    # ------------------------------------------------------------------------------------------------------------

    #print(match_found, match_counter, p_matches)
    if (match_found ==True and p_matches <10 and match_counter>5) or (match_found==True and len(limited_matches)<=2 and p_matches<400): #4 for other tests - # or c_question_duration > video_fps * float(line["C"+str(image_index+1)]):
        #print(match_found ==True and p_matches <10 and match_counter>5, c_question_duration > video_fps * float(line["C"+str(image_index+1)]))
        match_found = False
        #image_times_list.append([str(user_index), str(image_index), str(frame_counter)])
        #@if image_index ==0:
        #	print(",".join([str(user_index), str(image_index-1), str(question_start)]))
        #print(",".join([str(user_index), str(image_index), str(frame_counter)]))
        image_index+=1
        match_counter=0
    return match_found, match_counter, image_index

def adjust(fk_q, frame_counter, match_found, image_index):
    if "other_mc07_01_compressed_quadrants" in fk_q:
        if 330 <= frame_counter <= 362:
            return True, 0
        elif 786 <= frame_counter <= 828:
            return True, 14
    elif "other_mc03_01_compressed_quadrants" in fk_q:
        if 325 <= frame_counter <= 376:
            return True, 2
    elif "sespwalkup_11_1_01_camera_screen_compressed_quadrants" in fk_q:
        if 853 <= frame_counter <= 889:
            return True, 15
    elif "other_mc03_02_compressed_quadrants" in fk_q:
        if 393 <= frame_counter <= 434:
            return True, 4
    return match_found, image_index

def make_videos(fk_q, video_fps, video_len):
    videos = []
    path = "..\\test_results\\quiz\\{}".format(fk_q)
    if not os.path.exists(path):
        os.mkdir(path)
    name = os.path.join(path, "empty.mp4")
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    empty = cv.VideoWriter(name, fourcc, video_fps, (1280, 720))

    for i in range(0, video_len):
        name = os.path.join(path, str(i) + ".mp4")
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        videos.append(cv.VideoWriter(name, fourcc, video_fps, (1280, 720)))

    return videos, empty

def write_to_vid(img, match_found, lc_x, lc_y, frame_counter, empty, cnts, videos, image_index, quad_list):
    quad_list = list(set(quad_list))
    for i in range(0, len(lc_x)):
        img = cv.circle(img, (int(lc_x[i]), int(lc_y[i])), 5, (255, 0, 0))
    cv.drawContours(img, cnts, -1, (0, 255, 0), 3)
    cv.putText(img, "Frame: " + str(frame_counter), (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    if len(quad_list) != 0:
        cv.putText(img, ",".join(map(str, quad_list)) + " were added to the array at index " + str(image_index),
                   (0, 45), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    if match_found:
        cv.putText(img, "Image {} was detected".format(image_index), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                   (0, 0, 255))
        videos[image_index].write(img)
    if not match_found:
        empty.write(img)

nq_image_files, nq_images, nq_image_keypoints = get_images_and_keypoints("qualtrics_mapping_excluding_questions.csv")
f_image_files, f_images, f_image_keypoints = get_images_and_keypoints("qualtrics_mapping_combined.csv")

''' This loop iterates over the video files and performs feature extraction on each frame of the video.
It also opens the corresponding gaze file and processes the data according to the associated video frame
'''
survey_file = "data_output_excluding_questions.csv"
survey_csv = open(survey_file)
csv_file =csv.DictReader(survey_csv)

quadrants = {}

#add fixation revisits 
for line in csv_file:
    video_file, gaze_file = get_files(line) 
    if video_file is None or gaze_file is None: 
        continue

    dist_thresh = int(line["dist_thresh"])
    l_matches = int(line["l_matches"])

    if "lab" in video_file or "other" in video_file: 
        image_files, images, image_keypoints = f_image_files, f_images, f_image_keypoints 
    else: 
        image_files, images, image_keypoints = nq_image_files, nq_images, nq_image_keypoints 

    folder_key = os.path.splitext(video_file)[0].replace("\\", "_").replace("..", "")
    fk_q = folder_key + "_quadrants"
    quadrants[fk_q] = {}
    print(fk_q)

    # ------------------------------------------------------------------------------------------------------------
    # Opening file and gazedata
    # ------------------------------------------------------------------------------------------------------------
    cap = cv.VideoCapture(video_file)
    video_fps = float(cap.get(cv.CAP_PROP_FPS)) #5.0
    fsacc = process_gaze_datapoints(gaze_file, video_fps)

    videos, empty = make_videos(fk_q, video_fps, len(image_keypoints) + 1)

    # ------------------------------------------------------------------------------------------------------------
    # Initiatizing object detection 
    # ------------------------------------------------------------------------------------------------------------
    match_found = False
    image_index=0
    p_kp, p_dp = np.array([]),np.array([])
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
    ''' the loop below pull the frame from each video and then find the rectangle that contains the stimulus
    This is done by finding the contours, and looking for a contour of a specific size. Once that is found the x,y, height 
    and width to use as an offset later on
    '''
    #print("processing video", video_file)

    # ------------------------------------------------------------------------------------------------------------
    # Initializing state
    # ------------------------------------------------------------------------------------------------------------
    frame_counter = 0
    question_start = int(line["first_question"])*video_fps

    # TODO: Change back once environment testing is completed 
    # question_start = 149 * video_fps

    match_counter=0
    c_fixation_index=0
    ret = None 
    frame = None

    # ------------------------------------------------------------------------------------------------------------
    # Looking through video
    # ------------------------------------------------------------------------------------------------------------
    while (cap.isOpened()):
        c_fix_start_time, c_fix_end_time, lc_x, lc_y, c_fixation_index = get_point_and_time(fsacc, c_fixation_index, question_start, video_fps)

        # # ------------------------------------------------------------------------------------------------------------
        # # Getting data from fixations 
        # # c_x, c_y: I think are the last points at the end of fixation 
        # # ------------------------------------------------------------------------------------------------------------
        # c_fix_start_time, c_fix_end_time, c_fix_dur, p_x,p_y, c_x, c_y  = fsacc[c_fixation_index]

        # # ------------------------------------------------------------------------------------------------------------
        # # Skipping over data until question starts 
        # # ------------------------------------------------------------------------------------------------------------
        # # TODO: Change timestamp to when the portfolio emerges
        # while (int(c_fixation_index / 18 * video_fps) < question_start):
        #     c_fixation_index+=1
        #     c_fix_start_time, c_fix_end_time, c_fix_dur, p_x,p_y, c_x, c_y  = fsacc[c_fixation_index]
        while frame_counter not in range(int(c_fix_start_time), int(c_fix_end_time)):
            ret, frame=cap.read()
            frame_counter += 1

            # TODO: Change here too
            if frame_counter > 856:
                stop = 0
        
        # TODO: Normally, the code will always have a new frame with each fixation index so it will self terminate before 
        # the index goes out of bounds. In this case, it may not because the index can go out of bounds before a new frame 
        # is seen. Might just be the right move to terminate code when index is about to go out of bounds. 
        if c_fixation_index >= question_start + 30000: 
            break 

        # print(c_fix_start_time, c_fix_end_time, c_fixation_index, frame_counter, image_index)
        #print(frame_counter, question_start, ret)
        
        # ------------------------------------------------------------------------------------------------------------
        # Check if frame is available, picture is valid, and we are at the appropriate start time 
        # ------------------------------------------------------------------------------------------------------------
        if frame_counter >= question_start:
            if not (ret == True and image_index < len(image_files)): 
                break
            # TODO: update for consistency sake
            # if frame_counter in range(int(c_fix_start_time), int(c_fix_end_time)+1):
            #     c_fixation_index+=1

            #cv.imshow('Matches', frame)
            #print(frame_counter)
            #cv.waitKey()

            matches=None
               
            # Get contours
            img2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            img2 = cv.line(img2, (0, 683), (1280, 683), (255, 255, 255), 8)
            if "sespwalkup_11_1_01_camera_screen_compressed_quadrants" in fk_q:
                img2 = cv.line(img2, (0, 678), (1280, 678), (255, 255, 255), 7)
            _, thresh = cv.threshold(img2, 127, 255, cv.THRESH_BINARY_INV) #we use thresholding to find the bounding box
            contours, hier = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            # TODO: For testing
            # print(c_x, c_y)
            # with_dot = cv.circle(img2, (int(c_x), int(c_y)), 20, (255, 0, 0))
            # cv.imshow("with_dot", with_dot)
            # cv.waitKey(0)
            
            box_x, box_y, img_to_check, ret_cnt = get_box_of_interest(img2, contours)

            # ------------------------------------------------------------------------------------------------------------
            # Checking previous image with current image
            # ------------------------------------------------------------------------------------------------------------
            keypoints2, descriptors2 = detector.detectAndCompute(img_to_check, None)
            p_matches = get_pmatches(matcher, p_dp, descriptors2, dist_thresh)
            
            if type(descriptors2)!=None.__class__ and descriptors2.any():
                
                # ------------------------------------------------------------------------------------------------------------
                # Checking current image with all_stimuli image
                # ------------------------------------------------------------------------------------------------------------
                image_keypoint = image_keypoints[image_index]
                limited_matches = get_limited_matches(matcher, image_keypoint["descriptors"], descriptors2, dist_thresh)

                # ------------------------------------------------------------------------------------------------------------
                # box_x is the corner where the stimuli image starts in the video 
                # Subtracting it allows for c_x to be referenced from the origin of the stimuli image (given by all_stimuli)
                # Median offset is because the images are not perfectly similar (size or maybe cut off)
                # ------------------------------------------------------------------------------------------------------------
                x_median_offset, y_median_offset = get_median_offset(matcher, image_keypoint, descriptors2, keypoints2,
                                                                     dist_thresh)
                quad_list = []
                for xy_ind in range(0, len(lc_x)):
                    c_x = lc_x[xy_ind]
                    c_y = lc_y[xy_ind]
                    c_x_os, c_y_os = get_c_coords(c_x, c_y, x_median_offset - box_x, y_median_offset - box_y)
                    # quadrants = populate_quadrants(quadrants, images[image_index], c_x_os, c_y_os, fk_q, image_index)
                    quad_list = find_quadrants(quad_list, images[image_index], c_x_os, c_y_os)

                match_found, match_counter = check_match(limited_matches, match_counter, match_found, l_matches)
                match_found, match_counter, image_index = check_no_match(limited_matches, p_matches, match_counter,
                                                                         match_found, image_index)
                match_found, image_index = adjust(fk_q, frame_counter, match_found, image_index)

                if match_found:
                    quadrants = populate_quadrants(quadrants, fk_q, image_index, quad_list)

                p_kp,p_dp = keypoints2, descriptors2

                write_to_vid(frame, match_found, lc_x, lc_y, frame_counter, empty, ret_cnt, videos, image_index,
                             quad_list)
with open('..\\results\\quadrants.json', 'w', encoding='utf-8') as f:
    json.dump(quadrants, f, ensure_ascii=False, indent=4)

	#print(n_fixation_counter)
	#reaw_input()
	#csv_outfile = open(str(user_index)+"_gaze_output.csv","wb")
	#csv_out = csv.writer(csv_outfile, delimiter=",")
	#csv_out.writerows([[str(user_index), str(c_image_index), str(t),str(user_gaze_points[c_image_index][t][0]), str(user_gaze_points[c_image_index][t][1])] for c_image_index in user_gaze_points for t in user_gaze_points[c_image_index]])
	#csv_outfile.close()
	#user_index+=1
	#print("done processing video for", str(user_index))
#print(",".join(["user"]+[str(a).replace(",",";") for a in all_indices] + ["occ_"+str(a).replace(",",";") for a in all_indices] + ["avg_"+str(a).replace(",",";") for a in all_indices] + [str(a).replace(",",";") for a in n_fixation_counter_labels] + ["score","wrong"]))

# TODO: Uncomment this block
# csv_outfile = open("fixation_contour_occ_output_excluding_questions.csv","w")
# csv_out = csv.writer(csv_outfile, delimiter=",")
# csv_out.writerow(["user"]+[str(a).replace(",",";") for a in all_indices] + ["occ_"+str(a).replace(",",";") for a in all_indices if "x" not in a and "y" not in a] + ["avg_"+str(a).replace(",",";") for a in all_indices if "x" not in a and "y" not in a] + [str(a).replace(",",";") for a in n_fixation_counter_labels] + ["score","wrong"])
# #for user in user_fixation_occ:
# #	print(",".join([str(user)]+[str(user_fixation_counter[user][a]) for a in all_indices] + [str(user_fixation_occ[user][a]) for a in all_indices]+ [str(user_fixation_counter[user][a]/max(user_fixation_occ[user][a],1.0)) for a in all_indices] + [str(n_fixation_counter[user][a]) for a in n_fixation_counter_labels] + [str(user_data[user][0]),str(user_data[user][1])]))
# csv_out.writerows([[str(user)]+[str(user_fixation_counter[user][a]) for a in all_indices] + [str(user_fixation_occ[user][a]) for a in all_indices if "x" not in a and "y" not in a]+ [str(user_fixation_counter[user][a]/max(user_fixation_occ[user][a],1.0)) for a in all_indices if "x" not in a and "y" not in a] + [str(n_fixation_counter[user][a]) for a in n_fixation_counter_labels] + [str(user_data[user][0]),str(user_data[user][1])] for user in user_fixation_occ])
# csv_outfile.close()

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