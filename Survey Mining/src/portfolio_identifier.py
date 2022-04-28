from __future__ import print_function
import csv
import os
from scipy import stats
from detectors import *
import json
import cv2 as cv
import numpy as np

detector = cv.ORB_create(1000)


class referenceImage():

    def __init__(self, file_name):
        self.file_name = None
        self.img = None
        self.keypoints = None
        self.update_image_information(file_name)

        # Key is file name, value is contour path
        self.quadrant = {}

        self.match_found = False
        self.match_counter = 0
        self.kp = np.array([])
        self.dp = np.array([])

        self.p_matches = 0
        self.l_matches = []

        self.activity_log = []
        self.videos = {}

    def update_image_information(self, file_name):
        self.file_name = file_name
        c_img = cv.imread(file_name, 1)
        self.img = cv.cvtColor(c_img, cv.COLOR_BGR2GRAY)
        kp, dp = detector.detectAndCompute(self.img, None)
        self.keypoints = {"keypoints": kp, "descriptors": dp}

    def make_videos(self, key, video_fps, i):
        path = "..\\test_results\\building\\{}".format(key)
        if not os.path.exists(path):
            os.mkdir(path)
        name = os.path.join(path, str(i) + ".mp4")
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(name, fourcc, video_fps, (1280, 720))
        self.videos[key] = out

    def check_match(self):
        # ------------------------------------------------------------------------------------------------------------
        # Check if at least 5 matches because not all are accurate
        # ------------------------------------------------------------------------------------------------------------
        # TODO: Might need to decrease to 3
        if len(self.l_matches) >= 1:
            self.match_found = True
            self.match_counter += 1

    def check_no_match(self):
        # ------------------------------------------------------------------------------------------------------------
        # match_found is a mode. If the mode is switched on and these other conditions apply, then that means there is
        # no match found anymore and the match_found should be false.
        # ------------------------------------------------------------------------------------------------------------

        # print(match_found, match_counter, p_matches)
        if (self.match_found == True and self.p_matches < 10 and self.match_counter > 5) or \
                (self.match_found == True and len(self.l_matches) <= 2 and self.p_matches < 400):
            self.match_found = False
            self.match_counter = 0

    def update_log(self, c_x_os, c_y_os):
        x = 0
        y = 0
        (h, w) = self.img.shape

        one = np.array([[x, y], [x, h // 2], [w // 2, h // 2], [w // 2, y]]).reshape((-1, 1, 2)).astype(np.int32)
        two = np.array([[x + w // 2, y], [x + w // 2, h // 2], [x + w, h // 2], [x + w, y]]).reshape((-1, 1, 2)).astype(
            np.int32)
        three = np.array([[x, y + h // 2], [x, y + h], [x + w // 2, h], [x + w // 2, y + h // 2]]).reshape(
            (-1, 1, 2)).astype(np.int32)
        four = np.array([[x + w // 2, y + h // 2], [x + w // 2, y + h], [w, h], [x + w, y + h // 2]]).reshape(
            (-1, 1, 2)).astype(np.int32)

        # sh = cv.circle(c_img, (c_x_os, c_y_os), 20, (255, 0, 0))
        # sh = cv.rectangle(sh, (x, y), (w//2, h//2), (255, 0, 0))
        # cv.imshow("sh", sh)
        # cv.waitKey(0)

        if cv.pointPolygonTest(one, (c_x_os, c_y_os), 1) >= -5:
            self.activity_log.append(1)
        elif cv.pointPolygonTest(two, (c_x_os, c_y_os), 1) >= -5:
            self.activity_log.append(2)
        elif cv.pointPolygonTest(three, (c_x_os, c_y_os), 1) >= -5:
            self.activity_log.append(3)
        elif cv.pointPolygonTest(four, (c_x_os, c_y_os), 1) >= -5:
            self.activity_log.append(4)

    # c_img: images[image_index]
    def populate_quadrants(self, fk_q):
        self.activity_log = list(set(self.activity_log))
        for quad in self.activity_log:
            self.quadrant[fk_q].append(quad)


def add_points_to_log(r, lc_x, lc_y, x_median_offset, y_median_offset, box_x_start, box_y_start):
    for xy_ind in range(0, len(lc_x)):
        c_x = lc_x[xy_ind]
        c_y = lc_y[xy_ind]
        c_x_os, c_y_os = get_c_coords(c_x, c_y, x_median_offset - box_x_start, y_median_offset - box_y_start)
        if r.match_found:
            r.update_log(c_x_os, c_y_os)


def adjustments(ref_image_side_range, refs, video_file, l_image_files):
    if "mc03\\10_21" in video_file:
        ref_image_side_range = (600, 1280)

    if "sespwalkup\\11_1" in video_file:
        for i in range(0, 3):
            refs[i].update_image_information(l_image_files[i])
    elif "mc03\\10_21" not in video_file:
        for i in range(0, 4):
            refs[i].update_image_information(l_image_files[i])
    return ref_image_side_range


def clear_activity_logs(refs):
    for r in refs:
        r.activity_log = []


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

    if parent_folder == "other" or parent_folder == "lab":
        return None, None

    # TODO: Uncomment, when done with debugging
    # if not (parent_folder == "mc07" and folder_name == "11_2"):
    #     return None, None

    # ------------------------------------------------------------------------------------------------------------
    # Just a number because SSI labels the outputs as numbers, sometimes multiple files with
    # different numbers will be on there due to testing issues and SSI running multiple times
    # Each different number is a different time that SSI was ran
    # ------------------------------------------------------------------------------------------------------------
    file_ = line["file_name"]
    folder = os.path.join("..\\videos", os.path.join(parent_folder, folder_name))

    # ------------------------------------------------------------------------------------------------------------
    # Get corresponding video and gaze file based on the path given in the csv and the number the file has
    # ------------------------------------------------------------------------------------------------------------
    video_files = [os.path.join(folder, f) for f in os.listdir(folder) if ".mp4" in f]
    gaze_files = [os.path.join(folder, f) for f in os.listdir(folder) if "gazedata.stream~" in f]

    # ------------------------------------------------------------------------------------------------------------
    # Get file
    # ------------------------------------------------------------------------------------------------------------
    if file_ == "":
        video_file = video_files[0]
        gaze_file = gaze_files[0]
    else:
        video_file = [f for f in video_files if file_ in f][0]
        gaze_file = [f for f in gaze_files if file_ in f][0]
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
        g_x, g_y, t = points
        if float(g_x) > 1280:
            g_x = 0.0
        if float(g_y) > 800:
            g_y = 0.0
        g_t = i * 1000.0 / 90
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


def get_pmatches(matcher, p_dp, descriptors2):
    # This section looks at the level of similarity between the current frame and the previous frame.
    # This helps us know when the stimulus has changed.
    p_matches = 0
    if type(p_dp) != None.__class__ and p_dp.any():
        if type(descriptors2) != None.__class__ and descriptors2.any():
            matches = matcher.match(p_dp, descriptors2)
            # TODO: Maybe increase to 50
            # The value of 35 was selected to account for some slight differences and seems to work well in practice
            p_matches = len([match for match in matches if
                             match.distance < 65])
        else:
            p_matches = 0
    return p_matches


def get_limited_matches(matcher, image_descriptor, descriptors2):
    matches = matcher.match(image_descriptor, descriptors2)
    limited_matches = []  # not all of the matches are accurate. limited matches
    # We keep track of the similarity between the reference image and the current stimulus image

    # ------------------------------------------------------------------------------------------------------------
    # Get offsets to create mapping from gaze data in video to the reference image from all_stimuli
    # We are doing this because the analysis is done on the all_stimuli image rather than the current frame
    # ------------------------------------------------------------------------------------------------------------
    for match in matches:
        # TODO: Maybe increase to 50
        if match.distance < 65:  # 35
            limited_matches.append(match)

    return limited_matches


def get_median_offset(matcher, image_keypoint, descriptors2, keypoints2):
    matches = matcher.match(image_keypoint["descriptors"], descriptors2)
    keypoints1 = image_keypoint["keypoints"]
    keypoints_offsets = {'x': [], 'y': []}

    for match in matches:
        # TODO: Maybe increase to 50
        if match.distance < 65:  # 35
            keypoints_offsets['x'].append(int(keypoints1[match.queryIdx].pt[0]) - keypoints2[match.trainIdx].pt[
                0])  # these lists keep track of distance between points of correspondance of the reference and frame
            keypoints_offsets['y'].append(int(keypoints1[match.queryIdx].pt[1]) - keypoints2[match.trainIdx].pt[
                1])  # these lists keep track of distance between points of correspondance of the reference and frame
    x_median_offset = 0
    y_median_offset = 0
    # once we have the list of all corresponding points, we get the mode different, which practically corresponds to the most accurate mapping
    if len(stats.mode(np.array(keypoints_offsets['x'])).mode) > 0:
        x_median_offset = stats.mode(np.array(keypoints_offsets['x'])).mode[0]
        y_median_offset = stats.mode(np.array(keypoints_offsets['y'])).mode[0]
    return x_median_offset, y_median_offset


def get_c_coords(c_x, c_y, x_offset, y_offset):
    c_x = int(float(c_x))
    c_y = int(float(c_y))
    c_x_os = int(c_x + x_offset)
    c_y_os = int(c_y + y_offset)
    return c_x_os, c_y_os


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


def make_empty(fk_q, video_fps):
    name = os.path.join("..\\test_results\\building\\{}".format(fk_q), "empty.mp4")
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    empty = cv.VideoWriter(name, fourcc, video_fps, (1280, 720))
    return empty


def set_matches(refs, max_start, img_kps, l_matches_matrix, p_matches_matrix):
    raster_ind = 0
    for i in range(0, len(refs)):
        if i in range(max_start, max_start + len(img_kps)):
            refs[i].l_matches = l_matches_matrix[raster_ind][i]
            refs[i].p_matches = p_matches_matrix[raster_ind][i]
            raster_ind += 1
        else:
            refs[i].l_matches = []
            refs[i].p_matches = 0


def update_heuristic(r, ref_image_side_range, lbox_x, i):
    if r.match_found:
        if ref_image_side_range[1] - ref_image_side_range[0] > 150:
            ref_image_side_range = (lbox_x[i] - 75, lbox_x[i] + 75)
    return ref_image_side_range


def write_to_vid(img, refs, lc_x, lc_y, frame_counter, empty, fk_q, cnts, cnts_db, kWinName=None):
    for i in range(0, len(lc_x)):
        img = cv.circle(img, (int(lc_x[i]), int(lc_y[i])), 5, (255, 0, 0))
    matches = []
    offset = 0
    cv.drawContours(img, cnts, -1, (0, 255, 0), 3)

    # Blue is for rejected contours
    cv.drawContours(img, cnts_db, -1, (0, 0, 255), 3)
    for i in range(0, len(refs)):
        if refs[i].match_found:
            matches.append(str(i))
        if len(refs[i].activity_log) != 0:
            cv.putText(img, ",".join(map(str, refs[i].activity_log)) + " were added to the array for image index " +
                       str(i), (0, 15 * (offset + 2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            offset += 1
    cv.putText(img, "Images that are detected: " + ",".join(matches), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5,
               (0, 0, 255))
    if kWinName is not None:
        cv.imshow(kWinName, img)
        # 200 is normal speed
        cv.waitKey(1)

    cv.putText(img, "Frame: " + str(frame_counter), (0, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    if len(matches) == 0:
        empty.write(img)
    for i in matches:
        refs[int(i)].videos[fk_q].write(img)


kWinName = "eval"
# TODO: When writing videos, this should be faster
kWinName = None
if kWinName is not None:
    cv.namedWindow(kWinName, cv.WINDOW_NORMAL)

image_files = sorted([os.path.join("..\\portfolio_stimuli", "{}.png".format(i)) for i in range(1, 5)])
l_image_files = sorted([os.path.join("..\\portfolio_stimuli", "{}_left.png".format(i)) for i in range(1, 5)])
refs = []
for filename in image_files:
    refs.append(referenceImage(filename))

''' This loop iterates over the video files and performs feature extraction on each frame of the video.
It also opens the corresponding gaze file and processes the data according to the associated video frame
'''
survey_file = "data_output_excluding_questions.csv"
survey_csv = open(survey_file)
csv_file = csv.DictReader(survey_csv)

ref_offset = {0: 57, 1:0, 2:44, 3:36}

# add fixation revisits
for line in csv_file:
    video_file, gaze_file = get_files(line)
    if video_file is None or gaze_file is None:
        continue

    # Heuristic to narrow down which side the ref images are on when looking at the computer screen.
    ref_image_side_range = (0, 320)
    ref_image_side_range = adjustments(ref_image_side_range, refs, video_file, l_image_files)

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
            if not (ret == True and image_index < len(image_files)):
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

            p_matches_matrix = [list(map(lambda iv: get_pmatches(matcher, refs[iv[0]].dp, iv[1][1]), enumerate(row)))
                                for row in img_kps]
            l_matches_matrix = [list(map(lambda iv: get_limited_matches(matcher, refs[iv[0]].keypoints["descriptors"],
                                                                        iv[1][1]), enumerate(row))) for row in img_kps]

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

                x_median_offset, y_median_offset = get_median_offset(matcher, r.keypoints, descriptors2, keypoints2)
                add_points_to_log(r, lc_x, lc_y, x_median_offset, y_median_offset, lbox_x[i], lbox_y[i])
                r.populate_quadrants(fk_q)

            write_to_vid(frame, refs, lc_x, lc_y, frame_counter, empty, fk_q, cnts, cnts_db, kWinName)

for i in range(0, len(refs)):
    with open('..\\results\\building\\quadrants_{}.json'.format(i), 'w', encoding='utf-8') as f:
        json.dump(refs[i].quadrant, f, ensure_ascii=False, indent=4)
