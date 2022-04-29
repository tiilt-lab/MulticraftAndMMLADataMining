import numpy as np
from scipy import stats


def get_median_offset(matcher, image_keypoint, descriptors2, keypoints2, dist_thresh):
    matches = matcher.match(image_keypoint["descriptors"], descriptors2)
    keypoints1 = image_keypoint["keypoints"]
    keypoints_offsets = {'x': [], 'y': []}

    for match in matches:
        # TODO: Maybe increase to 50
        if match.distance < dist_thresh:  # 35
            keypoints_offsets['x'].append(int(keypoints1[match.queryIdx].pt[0]) - keypoints2[match.trainIdx].pt[0])  
            # these lists keep track of distance between points of correspondance of the reference and frame
            keypoints_offsets['y'].append(int(keypoints1[match.queryIdx].pt[1]) - keypoints2[match.trainIdx].pt[1]) 
            # these lists keep track of distance between points of correspondance of the reference and frame
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