import cv2 as cv
from .image_code.image_detection import detector

######################################
# portfolio_identifier.py specific   #
######################################

ref_offset = {0: 57, 1:0, 2:44, 3:36}

def adjustments(refs, video_file, l_image_files):
    if "sespwalkup\\11_1" in video_file:
        for i in range(0, 3):
            refs[i].update_image_information(l_image_files[i], detector)
    elif "mc03\\10_21" not in video_file:
        for i in range(0, 4):
            refs[i].update_image_information(l_image_files[i], detector)
    
    if "mc03\\10_21" in video_file:
        return (600, 1280)
    return (0, 320)

def update_heuristic(r, ref_image_side_range, lbox_x, i):
    if r.match_found:
        if ref_image_side_range[1] - ref_image_side_range[0] > 150:
            ref_image_side_range = (lbox_x[i] - 75, lbox_x[i] + 75)
    return ref_image_side_range


######################################
# quadrant_gaze_detector.py specific #
######################################

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

def match_image_list(video_file, fif, fi, fik, nif, ni, nik): 
    if "lab" in video_file or "other" in video_file: 
        return fif, fi, fik
    return nif, ni, nik 

def line_adjustment(img2, fk_q): 
    if "sespwalkup_11_1_01_camera_screen_compressed_quadrants" in fk_q:
        return cv.line(img2, (0, 678), (1280, 678), (255, 255, 255), 7)
    return img2