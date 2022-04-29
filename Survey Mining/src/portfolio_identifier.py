import os
import json
import cv2 as cv
from test_code.testing import *
from util.custom_adjustments import *
from util.gaze_detection import *
from util.input_file_processing import *
from util.image_code.image_detection import *
from util.image_code.point_calibration import *
from util.image_code.referenceImage import *


def add_points_to_log(r, lc_x, lc_y, x_median_offset, y_median_offset, box_x_start, box_y_start):
    for xy_ind in range(0, len(lc_x)):
        c_x = lc_x[xy_ind]
        c_y = lc_y[xy_ind]
        c_x_os, c_y_os = get_c_coords(c_x, c_y, x_median_offset - box_x_start, y_median_offset - box_y_start)
        if r.match_found:
            r.update_log(c_x_os, c_y_os)


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

refs = []
for filename in r_image_files:
    refs.append(referenceImage(filename, detector))

''' This loop iterates over the video files and performs feature extraction on each frame of the video.
It also opens the corresponding gaze file and processes the data according to the associated video frame
'''

# add fixation revisits
for line in csv_file:
    video_file, gaze_file = get_files(line, lambda pf, _: pf == "other" or pf == "lab")
    if video_file is None or gaze_file is None:
        continue

    # Heuristic to narrow down which side the ref images are on when looking at the computer screen.
    ref_image_side_range = adjustments(refs, video_file, l_image_files)

    folder_key = os.path.splitext(video_file)[0].replace("\\", "_").replace("..", "")
    fk_q = folder_key + "_quadrants"

    for r in refs:
        r.quadrant[fk_q] = []
    print(fk_q)

    cap = cv.VideoCapture(video_file)
    video_fps = float(cap.get(cv.CAP_PROP_FPS))  # 5.0
    fsacc = process_gaze_datapoints(gaze_file, video_fps)

    for i in range(0, len(refs)):
        refs[i].make_videos(fk_q, video_fps, i)

    empty = make_empty(fk_q, video_fps)

    image_index = 0
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
    ''' the loop below pull the frame from each video and then find the rectangle that contains the stimulus
    This is done by finding the contours, and looking for a contour of a specific size. Once that is found the x,y, 
    height and width to use as an offset later on
    '''
    # print("processing video", video_file)

    frame_counter = 0
    question_start = int(line["build_start"]) * video_fps

    # TODO: Change back once environment testing is completed
    # 470 * 5 for cnt 2
    # 692 * 5 for cnt 3 & 4
    # 149 * 5 for beginning
    # question_start = 692 * 5
    # question_start = int(line["build_start"]) * video_fps

    c_fixation_index = 0
    ret = None
    frame = None

    # ------------------------------------------------------------------------------------------------------------
    # Looking through video
    # ------------------------------------------------------------------------------------------------------------
    while (cap.isOpened()):
        c_fix_start_time, c_fix_end_time, lc_x, lc_y, c_fixation_index = get_point_and_time(fsacc, c_fixation_index,
                                                                                            question_start, video_fps)

        clear_activity_logs(refs)

        # # ------------------------------------------------------------------------------------------------------------
        # # Getting data from fixations
        # # c_x, c_y: I think are the last points at the end of fixation
        # # ------------------------------------------------------------------------------------------------------------
        # c_fix_start_time, c_fix_end_time, c_fix_dur, p_x,p_y, c_x, c_y  = fsacc[c_fixation_index]

        while frame_counter not in range(int(c_fix_start_time), int(c_fix_end_time)):
            ret, frame = cap.read()
            frame_counter += 1
            # if frame_counter % 100 == 0:
            #     print(frame_counter)

        # Normally, the code will always have a new frame with each fixation index so it will self terminate before
        # the index goes out of bounds. In this case, it may not because the index can go out of bounds before a new
        # frame is seen. Might just be the right move to terminate code when index is about to go out of bounds.

        if c_fixation_index >= len(fsacc):
            break

        # ------------------------------------------------------------------------------------------------------------
        # Check if frame is available, picture is valid, and we are at the appropriate start time
        # ------------------------------------------------------------------------------------------------------------
        # Sometimes gaze data be missing from question_start, so we start gaze data at nearest frame from question start
        if frame_counter >= question_start:
            if not (ret == True and image_index < len(r_image_files)):
                break

            # Get contours
            img2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _, thresh = cv.threshold(img2, 127, 255,
                                     cv.THRESH_BINARY_INV)  # we use thresholding to find the bounding box
            contours, hier = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            # For testing
            # print(c_x, c_y)
            # with_dot = cv.circle(img2, (int(c_x), int(c_y)), 20, (255, 0, 0))
            # cv.imshow("with_dot", with_dot)
            # cv.waitKey(0)

            # TODO: Figure out last image -- maybe make more naive, but need it to work in general
            # TODO: Make heuristic for side of ref images
            # if frame_counter >= 692 * 5:
            #     stop = 0

            #       1    2    3    4
            # img1
            # img2
            # img3

            # if frame_counter > 1200:
            #     stop = 0
            box_info, cnts, cnts_db = get_boxes_of_interest(img2, contours, ref_image_side_range)
            lbox_x = [x for x, _, _ in box_info]
            lbox_y = [y for _, y, _ in box_info]

            img_kps = get_options_matrix(box_info, ref_offset)

            p_matches_matrix = [list(map(lambda iv: get_pmatches(matcher, refs[iv[0]].dp, iv[1][1], 65), enumerate(row)))
                                for row in img_kps]
            l_matches_matrix = [list(map(lambda iv: get_limited_matches(matcher, refs[iv[0]].keypoints["descriptors"],
                                                                        iv[1][1], 65), enumerate(row))) for row in img_kps]

            max_start = get_window(img_kps, refs, l_matches_matrix)

            set_matches(refs, max_start, img_kps, l_matches_matrix, p_matches_matrix)

            img_kps = [img_kps[i][max_start + i] for i in range(0, len(img_kps))]
            lbox_x = [lbox_x[i] + ref_offset[max_start + i] for i in range(0, len(lbox_x))]

            # ------------------------------------------------------------------------------------------------------------
            # box_x is the corner where the stimuli image starts in the video
            # Subtracting it allows for c_x to be referenced from the origin of the stimuli image (given by all_stimuli)
            # Median offset is because the images are not perfectly similar (size or maybe cut off)
            # ------------------------------------------------------------------------------------------------------------
            for r in refs:
                r.check_match()
                r.check_no_match()

            for i in range(0, len(img_kps)):
                keypoints2 = img_kps[i][0]
                descriptors2 = img_kps[i][1]
                r = refs[max_start + i]
                r.kp = keypoints2
                r.dp = descriptors2

                ref_image_side_range = update_heuristic(r, ref_image_side_range, lbox_x, i)

                x_median_offset, y_median_offset = get_median_offset(matcher, r.keypoints, descriptors2, keypoints2, 65)
                add_points_to_log(r, lc_x, lc_y, x_median_offset, y_median_offset, lbox_x[i], lbox_y[i])
                r.populate_quadrants(fk_q)

            write_to_vid_building(frame, refs, lc_x, lc_y, frame_counter, empty, fk_q, cnts, cnts_db, kWinName)

for i in range(0, len(refs)):
    with open('..\\results\\building\\quadrants_{}.json'.format(i), 'w', encoding='utf-8') as f:
        json.dump(refs[i].quadrant, f, ensure_ascii=False, indent=4)
