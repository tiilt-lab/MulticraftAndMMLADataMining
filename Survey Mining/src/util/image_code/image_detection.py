import cv2 as cv 

detector = cv.ORB_create(1000)

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


def get_limited_matches(matcher, image_descriptor, descriptors2, dist_thresh):
    matches = matcher.match(image_descriptor, descriptors2)
    limited_matches = []  # not all of the matches are accurate. limited matches
    # We keep track of the similarity between the reference image and the current stimulus image

    # ------------------------------------------------------------------------------------------------------------
    # Get offsets to create mapping from gaze data in video to the reference image from all_stimuli
    # We are doing this because the analysis is done on the all_stimuli image rather than the current frame
    # ------------------------------------------------------------------------------------------------------------
    for match in matches:
        # TODO: Maybe increase to 50
        if match.distance < dist_thresh:  # 35
            limited_matches.append(match)

    return limited_matches

######################################
# portfolio_identifier.py specific   #
######################################

def get_window(img_kps, refs, l_matches_matrix):
    if len(img_kps) > len(refs):
        raise Exception("More photos than refs")
    window = list(range(0, len(img_kps)))
    max_start = 0
    max_val = 0
    while len(window) > 0 and window[-1] < len(refs):
        raster_ind = 0
        val = 0
        for wind_ind in window:
            val += len(l_matches_matrix[raster_ind][wind_ind])
            raster_ind += 1
        if val > max_val:
            max_start = window[0]
            max_val = val
        nxt = window[0] + 1
        window = list(range(nxt, nxt + len(window))) if len(window) > 1 else [window[0] + 1]
    return max_start


def get_options_matrix(box_info, ref_offset):
    img_kps = []
    for x, y, img_to_check in box_info:
        row = []
        for offset_ind in range(0, 4):
            offset = ref_offset[offset_ind]
            keypoints2, descriptors2 = detector.detectAndCompute(img_to_check[offset:, :], None)
            if not (type(descriptors2) != None.__class__ and descriptors2.any()):
                row = None
                continue
            row.append([keypoints2, descriptors2])
        if row is not None:
            img_kps.append(row)
    return img_kps

def get_boxes_of_interest(img2, contours, cnt_heuristic):
    ret = []
    cnts = []
    cnts_db = []

    # ------------------------------------------------------------------------------------------------------------
    # Look at each contour
    # ------------------------------------------------------------------------------------------------------------
    for cnt in contours:
        (x, y, w, h) = cv.boundingRect(cnt)
        # print(x,y,w,h)

        # ------------------------------------------------------------------------------------------------------------
        # Check if contour is in the box
        # ------------------------------------------------------------------------------------------------------------
        # 500 was good for portfolio
        if (w > 500 and w < 800) and (h > 100 and h < 500):
            img_to_check = img2.copy()
            img_to_check = img_to_check[max(0, y-57):y+h, x:x+w]

            if cnt_heuristic is None or cnt_heuristic[0] < x < cnt_heuristic[1]:
                ret.append([x, y, img_to_check])
                cnts.append(cnt)
            else:
                cnts_db.append(cnt)

    ret.sort(key=lambda z: z[1])
    return ret, cnts, cnts_db

######################################
# quadrant_gaze_detector.py specific #
######################################

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