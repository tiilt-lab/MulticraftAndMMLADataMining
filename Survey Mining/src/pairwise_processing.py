import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import array
from copy import deepcopy
from dtaidistance import dtw, dtw_ndim
from scipy.spatial.distance import euclidean
from textwrap import wrap


def change_all_lists(fn, dct, loqd):
    new_dct = change_list_in_stimuli_data(fn, dct)
    new_loqd = change_list_in_ref_data(fn, loqd)
    return new_dct, new_loqd


def change_list_in_ref_data(fn, loqd):
    ret = deepcopy(loqd)
    for i in range(0, len(ret)):
        for user in ret[i].keys():
            ret[i][user] = fn(ret[i][user])
        # If one wants to rewrite data
        # with open(files, 'w', encoding='utf-8') as quad_f:
        #     json.dump(quad_data, quad_f, ensure_ascii=False, indent=4)
    return ret


def change_list_in_stimuli_data(fn, dct):
    ret = deepcopy(dct)
    for user in usernames:
        question_keys = sorted(list(ret[user].keys()))
        for question_key in question_keys:
            ret[user][question_key] = fn(ret[user][question_key])
    return ret


def convert_all_to_heatmap(list_of_distance_matrices, axis):
    return list(map(lambda dm: make_heatmap(dm, axis), list_of_distance_matrices))


def convert_to_pairwise_matrices(list_of_data):
    for i in range(0, len(list_of_data)):
        matrix = [[None for _ in list_of_data[i]] for _ in list_of_data[i]]
        for j in range(0, len(list_of_data[i])):
            pair1 = list_of_data[i][j]
            for k in range(0, len(list_of_data[i])):
                pair2 = list_of_data[i][k]
                if isinstance(pair2[0], list):
                    dist = dtw_ndim.distance_fast(np.array(pair1, dtype=np.double), np.array(pair2, dtype=np.double))
                else:
                    dist = dtw.distance_fast(array.array('d', pair1), array.array('d', pair2))
                matrix[j][k] = dist
        list_of_data[i] = matrix


def direction(lst):
    # [1, 0, 0] is horizontal
    # [0, 1, 0] is vertical
    # [0, 0, 1] is diagonal
    lst = truncate(lst)
    ret = []
    for i in range(0, len(lst) - 1):
        diff = abs(lst[i] - lst[i + 1])
        if lst[i] in [2, 3] and lst[i + 1] in [2, 3] or abs(lst[i] - lst[i + 1]) == 3:
            ret.append([0, 0, 1])
        elif diff == 1:
            ret.append([1, 0, 0])
        elif diff == 2:
            ret.append([0, 1, 0])
        else:
            raise Exception("Unexpected result, diff: " + diff)
    return ret


# compare between the structures a person was building
def populate_within_builds(within_builds, loqd):
    unames = None
    for quad_data in loqd:
        unames = sorted(list(quad_data.keys()))
        for user_ind in range(0, len(unames)):
            within_builds[user_ind].append(quad_data[unames[user_ind]])
    return ["Comparing performance between builds for user " + user for user in unames]


# compare between the people for each structure
def populate_within_refs(within_refs, loqd):
    i = 0
    for quad_data in loqd:
        for user in quad_data.keys():
            within_refs[i].append(quad_data[user])
        i += 1
    return ["Comparing people's performance on build " + str(i) for i in range(0, 4)]


# compare between the people for each question
def populate_within_stimuli(within_stimuli, data):
    usernames = sorted(list(data.keys()))
    questions = None
    for user in usernames:
        questions = sorted(list(data[user].keys()))
        for i in range(0, len(questions)):
            within_stimuli[i].append(data[user][questions[i]])
    return ["Comparing people's performance on question " + question for question in questions]


# compare between the questions each person was doing
def populate_within_user(within_user, data):
    usernames = sorted(list(data.keys()))
    for user_ind in range(0, len(usernames)):
        questions = sorted(list(data[usernames[user_ind]].keys()))
        for i in range(0, len(questions)):
            within_user[user_ind].append(data[usernames[user_ind]][questions[i]])
    return ["Comparing performance between questions for user " + user for user in usernames]


def make_heatmap(arr, axis):
    ax = sns.heatmap(
        arr,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    plt.show()
    ax.set_xticklabels(
        axis,
        rotation=45,
        horizontalalignment='right'
    )
    ax.set_yticklabels(
        axis,
        rotation=0
    )
    return ax


def remove_extra_questions(usernames, dct):
    for user in usernames:
        replace = {}
        question_keys = sorted(list(dct[user].keys()))
        questions = dct[user]

        skip = [1, 18, 22, 23]
        if len(question_keys) == 24:
            new_question_ind = 0
            for i in range(0, 24):
                if i in skip:
                    continue
                replace[str(new_question_ind)] = questions[str(i)]
                new_question_ind += 1
        else:
            for i in range(0, 20):
                replace[str(i)] = questions[str(i)]
        dct[user] = replace


def truncate(lst):
    ret = []
    curr = None
    for val in lst:
        if curr != val:
            ret.append(val)
            curr = val
        else:
            continue
    return ret


def check_and_make_folder(parent, folder):
    path = os.path.join(parent, folder)
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def make_folders():
    for parent in [".\\comparing_people", ".\\comparing_images"]:
        if not os.path.exists(parent):
            os.mkdir(parent)
        for child in ["building", "quiz"]:
            new_path = check_and_make_folder(parent, child)
            for grandchildren in ["normal", "trunc", "direction"]:
                check_and_make_folder(new_path, grandchildren)


make_folders()
with open("..\\results\\quadrants.json") as f:
    data = json.load(f)

# del data["_videos_lab_mc04_02_camera_screen_quadrants"]

list_of_quad_data = []
for files in os.listdir('..\\results\\building'):
    with open(os.path.join('..\\results\\building', files)) as quad_f:
        quad_data = json.load(quad_f)
        list_of_quad_data.append(quad_data)

usernames = sorted(list(data.keys()))

# Clean data up -- in particular, remove the extra questions
remove_extra_questions(usernames, data)

# Extra processing of data
trunc_data, trunc_loqd = change_all_lists(truncate, data, list_of_quad_data)
direction_data, direction_loqd = change_all_lists(direction, data, list_of_quad_data)

# Creating user axis
user_axis = [user_name for user_name in usernames]
user_build_axis = [user_name for user_name in list_of_quad_data[0].keys()]

# Creating ref axis
ref_axis = [i for i in range(0, 4)]

# Creating question axis
question_axis = [i for i in range(1, 21)]


# pdmh: process_data_and_make_heatmap
def pdmh(quadrant_data, heatmap_arr, populate_fn, axis, parent, child):
    # Users will be the axis
    titles = populate_fn(heatmap_arr, quadrant_data)
    convert_to_pairwise_matrices(heatmap_arr)
    htmp_builds = convert_all_to_heatmap(heatmap_arr, axis)
    for i in range(0, len(htmp_builds)):
        htmp = htmp_builds[i]
        title = titles[i]
        htmp.set_title("\n".join(wrap(title, 60)))
        fname = title.split(" ")[-1]
        htmp.get_figure().savefig(parent + child + "\\" + fname + ".png", bbox_inches="tight")


# "trunc", "direction"
# loqd_dict = {"normal": list_of_quad_data}
# data_dict = {"normal": data}
loqd_dict = {"normal": list_of_quad_data, "trunc": trunc_loqd, "direction": direction_loqd}
data_dict = {"normal": data, "trunc": trunc_data, "direction": direction_data}
for child in loqd_dict.keys():
    list_of_quad_data = loqd_dict[child]
    data = data_dict[child]
    heatmap = [[] for user_name in list_of_quad_data[0].keys()]
    pdmh(list_of_quad_data, heatmap, populate_within_builds, ref_axis, ".\\comparing_images\\building\\", child)
    heatmap = [[] for i in range(0, 4)]
    pdmh(list_of_quad_data, heatmap, populate_within_refs, user_build_axis, ".\\comparing_people\\building\\", child)
    heatmap = [[] for i in range(0, 20)]
    pdmh(data, heatmap, populate_within_stimuli, user_axis, ".\\comparing_people\\quiz\\", child)
    heatmap = [[] for user_name in usernames]
    pdmh(data, heatmap, populate_within_user, question_axis, ".\\comparing_images\\quiz\\", child)

