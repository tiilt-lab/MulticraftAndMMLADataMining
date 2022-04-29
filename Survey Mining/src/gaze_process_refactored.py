import json
import os
import cv2 as cv
import numpy as np
from util.gaze_detection import *
from util.input_file_processing import *
from util.image_code.image_detection import *
from util.image_code.point_calibration import *
from util.image_code.referenceImage import *


def get_point_and_time(fsacc, c_fixation_index, question_start, video_fps): 
    c_fix_start_time, c_fix_end_time, c_fix_dur, p_x,p_y, c_x, c_y  = fsacc[c_fixation_index]
    c_fix_end_time/=1000.0
    c_fix_end_time*=video_fps
    c_fix_end_time=round(c_fix_end_time)
    c_fix_start_time/=1000.0
    c_fix_start_time*=video_fps
    c_fix_start_time = round(c_fix_start_time)

    # ------------------------------------------------------------------------------------------------------------
    # Skipping over data until question starts 
    # ------------------------------------------------------------------------------------------------------------
    while(int(c_fix_end_time) < question_start):
        c_fixation_index+=1
        c_fix_start_time, c_fix_end_time, c_fix_dur,p_x,p_y, c_x, c_y  = fsacc[c_fixation_index]
        c_fix_end_time/=1000.0
        c_fix_end_time*=video_fps
        c_fix_end_time = round(c_fix_end_time)
        c_fix_start_time/=1000.0
        c_fix_start_time*=video_fps
        c_fix_start_time = round(c_fix_start_time) 
    
    return c_fix_start_time, c_fix_end_time, c_x, c_y, c_fixation_index


image_files, images, image_keypoints = get_images_and_keypoints("..\\input_files\\qualtrics_mapping_excluding_questions.csv")

''' This loop iterates over the video files and performs feature extraction on each frame of the video.
It also opens the corresponding gaze file and processes the data according to the associated video frame
'''
quadrants = {}

#add fixation revisits 
for line in csv_file:
    video_file, gaze_file = get_files(line) 
    if video_file is None or gaze_file is None: 
        continue

    folder_key = os.path.splitext(video_file)[0].replace("\\", "_").replace("..", "")
    fk_q = folder_key + "_quadrants"
    quadrants[fk_q] = {}

    # ------------------------------------------------------------------------------------------------------------
    # Opening file and gazedata
    # ------------------------------------------------------------------------------------------------------------
    cap = cv.VideoCapture(video_file)
    video_fps = float(cap.get(cv.CAP_PROP_FPS)) #5.0
    fsacc = process_gaze_datapoints(gaze_file, video_fps, lambda w, x, y, z: saccade_detection(w, x, y, missing=0.0, minlen=5, maxvel=40, maxacc=340)[1])

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

    # ------------------------------------------------------------------------------------------------------------
    # Looking through video
    # ------------------------------------------------------------------------------------------------------------
    while (cap.isOpened()):
        c_fix_start_time, c_fix_end_time, c_x, c_y, c_fixation_index = get_point_and_time(fsacc, c_fixation_index, question_start, video_fps)

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
        ret, frame=cap.read()
        #print(frame_counter, question_start, ret)
        
        # ------------------------------------------------------------------------------------------------------------
        # Check if frame is available, picture is valid, and we are at the appropriate start time 
        # ------------------------------------------------------------------------------------------------------------
        if frame_counter >= question_start:
            if not (ret == True and image_index < len(image_files)): 
                break
            # TODO: update for consistency sake
            # c_fixation_index += 1 
            if frame_counter in range(int(c_fix_start_time), int(c_fix_end_time)+1):
                c_fixation_index+=1
            #cv.imshow('Matches', frame)
            #print(frame_counter)
            #cv.waitKey()

            matches=None
               
            # Get contours
            img2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(img2, 127, 255, cv.THRESH_BINARY_INV) #we use thresholding to find the bounding box
            contours, hier = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            # TODO: For testing
            # print(c_x, c_y)
            # with_dot = cv.circle(img2, (int(c_x), int(c_y)), 20, (255, 0, 0))
            # cv.imshow("with_dot", with_dot)
            # cv.waitKey(0)
            
            box_x, box_y, img_to_check, _ = get_box_of_interest(img2, contours)

            # ------------------------------------------------------------------------------------------------------------
            # Checking previous image with current image
            # ------------------------------------------------------------------------------------------------------------
            keypoints2, descriptors2 = detector.detectAndCompute(img_to_check, None)
            p_matches = get_pmatches(matcher, p_dp, descriptors2, 50)
            
            if type(descriptors2)!=None.__class__ and descriptors2.any():
                
                # ------------------------------------------------------------------------------------------------------------
                # Checking current image with all_stimuli image
                # ------------------------------------------------------------------------------------------------------------
                image_keypoint = image_keypoints[image_index]
                limited_matches = get_limited_matches(matcher, image_keypoint["descriptors"], descriptors2, 50)

                # ------------------------------------------------------------------------------------------------------------
                # box_x is the corner where the stimuli image starts in the video 
                # Subtracting it allows for c_x to be referenced from the origin of the stimuli image (given by all_stimuli)
                # Median offset is because the images are not perfectly similar (size or maybe cut off)
                # ------------------------------------------------------------------------------------------------------------
                x_median_offset, y_median_offset = get_median_offset(matcher, image_keypoint, descriptors2, keypoints2, 50)
                c_x_os, c_y_os = get_c_coords(c_x, c_y, x_median_offset - box_x, y_median_offset - box_y)

                quadrants = update_quadrants_dict(quadrants, images[image_index], c_x_os, c_y_os, fk_q, image_index)

                match_found, match_counter = check_match(limited_matches, match_counter, match_found, 3)
                match_found, match_counter, image_index = check_no_match(limited_matches, p_matches, match_counter, match_found, image_index)

                p_kp,p_dp = keypoints2, descriptors2
        frame_counter+=1

with open('..\\results\\quadrants_og.json', 'w', encoding='utf-8') as f:
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