import cv2 as cv
import numpy as np 
import os


def find_quadrants(c_img, c_x_os, c_y_os):
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
        return 1
    elif cv.pointPolygonTest(two, (c_x_os, c_y_os), 1) >= -5:
        return 2
    elif cv.pointPolygonTest(three, (c_x_os, c_y_os), 1) >= -5:
        return 3
    elif cv.pointPolygonTest(four, (c_x_os, c_y_os), 1) >= -5:
        return 4
    return None

######################################
# portfolio_identifier.py specific   #
######################################

class referenceImage():

    def __init__(self, file_name, detector):
        self.file_name = None
        self.img = None
        self.keypoints = None
        self.update_image_information(file_name, detector)

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

    def update_image_information(self, file_name, detector):
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
        quadrant = find_quadrants(self.img, c_x_os, c_y_os)
        if quadrant is not None: 
            self.activity_log.append(quadrant)

    # c_img: images[image_index]
    def populate_quadrants(self, fk_q):
        self.activity_log = list(set(self.activity_log))
        for quad in self.activity_log:
            self.quadrant[fk_q].append(quad)

def clear_activity_logs(refs):
    for r in refs:
        r.activity_log = []

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

######################################
# quadrant_gaze_detector.py specific #
######################################

def update_quadrants(ret, c_img, c_x_os, c_y_os):
    quadrant = find_quadrants(c_img, c_x_os, c_y_os)
    if quadrant is not None: 
        ret.append(quadrant)
    return ret

#######################################
# gaze_process_refactored.py specific #
#######################################

# c_img: images[image_index]
def update_quadrants_dict(quadrants, c_img, c_x_os, c_y_os, fk_q, image_index):
    if image_index not in quadrants[fk_q]: 
        quadrants[fk_q][image_index] = []
    quadrants[fk_q][image_index] = update_quadrants(quadrants[fk_q][image_index], c_img, c_x_os, c_y_os)
    return quadrants
