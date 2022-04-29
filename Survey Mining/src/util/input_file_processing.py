import os 
import numpy as np 
import csv 
import cv2 as cv 
from .gaze_detection import *
from .image_code.image_detection import detector

survey_file = "..\\input_files\\data_output_excluding_questions.csv"
survey_csv = open(survey_file)
csv_file = csv.DictReader(survey_csv)

def get_files(line, custom_condition=lambda x, y: False):
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

    if custom_condition(parent_folder, folder_name):
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


def process_gaze_datapoints(gaze_file, video_fps, callback=lambda w, x, y, z: filter_gaze(w, x, y, z)):
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

    return callback(np.array(gaze_x), np.array(gaze_y), np.array(gaze_time), video_fps)

######################################
# portfolio_identifier.py specific   #
######################################

r_image_files = sorted([os.path.join("..\\portfolio_stimuli", "{}.png".format(i)) for i in range(1, 5)])
l_image_files = sorted([os.path.join("..\\portfolio_stimuli", "{}_left.png".format(i)) for i in range(1, 5)])

######################################
# quadrant_gaze_detector.py specific #
######################################

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

nq_image_files, nq_images, nq_image_keypoints = get_images_and_keypoints("..\\input_files\\qualtrics_mapping_excluding_questions.csv")
f_image_files, f_images, f_image_keypoints = get_images_and_keypoints("..\\input_files\\qualtrics_mapping_combined.csv")