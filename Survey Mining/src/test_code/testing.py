import os 
import cv2 as cv
import json

building_path = "..\\..\\test_results\\building\\" 
quiz_path = "..\\..\\test_results\\quiz\\" 

def get_seconds_count(vid): 
    frames = 0 
    cap = cv.VideoCapture(vid) 
    video_fps = float(cap.get(cv.CAP_PROP_FPS))
    while (cap.isOpened()): 
        ret, _ = cap.read()
        if ret: 
            frames += 1 
        else: 
            break
    if video_fps == 0: 
        return 0
    return frames // video_fps


def make_data_ref():
    print("Making reference json")
    ref = {}
    build_files = [os.path.join(building_path, dir) for dir in os.listdir(building_path)]
    quiz_files = [os.path.join(quiz_path, dir) for dir in os.listdir(quiz_path)]

    for dir in build_files + quiz_files: 
        if os.path.isfile(dir): 
            continue
        ref[dir] = {} 
        for vid in os.listdir(dir): 
            print(os.path.join(dir, vid))
            seconds = get_seconds_count(os.path.join(dir, vid)) 
            ref[dir][vid] = seconds
    
    with open("..\\..\\test_results\\seconds_per_video.json", "w") as f: 
        json.dump(ref, f, ensure_ascii=False, indent=4)


def compare_data(): 
    log = open("..\\..\\test_results\\log.txt", "w")
    common = 0 
    original_extra = 0 
    new_extra = 0
    with open("..\\..\\test_results\\seconds_per_video.json") as f: 
        data = json.load(f) 
        build_files = [os.path.join(building_path, dir) for dir in os.listdir(building_path)]
        quiz_files = [os.path.join(quiz_path, dir) for dir in os.listdir(quiz_path)]
        for dir in build_files + quiz_files: 
            if os.path.isfile(dir): 
                continue
            for vid in os.listdir(dir): 
                long_path = os.path.join(dir, vid)
                seconds = get_seconds_count(long_path) 
                if vid not in data[dir].keys(): 
                    error_msg = "{} not in data.".format(long_path)
                    print(error_msg) 
                    log.write(error_msg + "\n")
                    new_extra += 1
                    continue 
                common += 1
                if data[dir][vid] == seconds: 
                    print("{} passed test".format(long_path))
                else: 
                    fail_msg = "{} failed test by {} seconds".format(long_path, abs(data[dir][vid] - seconds))
                    print(fail_msg) 
                    log.write(fail_msg + "\n") 
        original_extra = count_vids(data) - common
        log.write("{} different videos are from the original and {} different videos are from the new videos.\n".format(original_extra, new_extra))


def count_vids(data): 
    count = 0
    for i in data.keys(): 
        count += len(data[i].keys()) 
    return count

######################################
# portfolio_identifier.py specific   #
######################################

kWinName = "eval"
# TODO: When writing videos, this should be faster
kWinName = None
if kWinName is not None:
    cv.namedWindow(kWinName, cv.WINDOW_NORMAL)

def make_empty(fk_q, video_fps):
    name = os.path.join(building_path + fk_q, "empty.mp4")
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    empty = cv.VideoWriter(name, fourcc, video_fps, (1280, 720))
    return empty

def write_to_vid_building(img, refs, lc_x, lc_y, frame_counter, empty, fk_q, cnts, cnts_db, kWinName=None):
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

######################################
# quadrant_gaze_detector.py specific #
######################################

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

def write_to_vid_quiz(img, match_found, lc_x, lc_y, frame_counter, empty, cnts, videos, image_index, quad_list):
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

if __name__ == "__main__": 
    if not os.path.exists("..\\..\\test_results\\seconds_per_video.json"):
        make_data_ref()
    compare_data() 
