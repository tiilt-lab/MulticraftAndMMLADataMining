import seaborn as sns
import json
import os
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from copy import deepcopy
import matplotlib.pyplot as plt


def change_all_lists(fn, dct, loqd):
    new_dct = change_list_in_stimuli_data(fn, dct)
    new_loqd = change_list_in_ref_data(fn, loqd)
    return new_dct, new_loqd


def change_list_in_ref_data(fn, loqd):
    ret = deepcopy(loqd)
    for i in range(0, len(ret)):
        for user in usernames:
            ret[i][user] = fn(ret[i][user])
        # If one wants to rewrite data
        # with open(files, 'w', encoding='utf-8') as quad_f:
        #     json.dump(quad_data, quad_f, ensure_ascii=False, indent=4)
    return ret


def change_list_in_stimuli_data(fn, dct):
    ret = deepcopy(dct)
    for user in usernames:
        question_keys = sorted(list(user.keys()))
        for question_key in question_keys:
            ret[user][question_key] = fn(ret[user][question_key])
    return ret


def convert_all_to_heatmap(list_of_distance_matrices, axis):
    return list(map(lambda dm: make_heatmap(dm, axis), list_of_distance_matrices))


def convert_to_pairwise_matrices(list_of_data):
    for i in range(0, len(list_of_data)):
        matrix = [[None for _ in list_of_data] for _ in list_of_data]
        for j in range(0, len(list_of_data[i])):
            pair1 = list_of_data[i][j]
            for k in range(0, len(list_of_data[i])):
                pair2 = list_of_data[i][k]
                dist, path = fastdtw(np.array(pair1), np.array(pair2), dist=euclidean)
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


def populate_within_builds(usernames, within_refs):
    for files in os.listdir('.\\building_ref_data'):
        for user_ind in range(0, len(usernames)):
            with open(files) as quad_f:
                quad_data = json.load(quad_f)
            within_refs[user_ind].append(quad_data[usernames[user_ind]])


def populate_within_refs(usernames, within_refs):
    i = 0
    for files in os.listdir('.\\building_ref_data'):
        for user in usernames:
            with open(files) as quad_f:
                quad_data = json.load(quad_f)
            within_refs[i].append(quad_data[user])
        i += 1


def populate_within_stimuli(usernames, within_stimuli):
    for user in usernames:
        questions = sorted(list(user.keys()))
        for i in range(0, len(questions)):
            within_stimuli[i].append(data[user][questions[i]])


def populate_within_user(usernames, within_user):
    for user_ind in range(0, len(usernames)):
        questions = sorted(list(usernames[user_ind].keys()))
        for i in range(0, len(questions)):
            within_user[user_ind].append(data[usernames[user_ind]][questions[i]])


def make_heatmap(arr, axis):
    ax = sns.heatmap(
        arr,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
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
        question_keys = sorted(list(user.keys()))
        questions = dct[user]

        skip = [1, 18, 22, 23]
        if len(question_keys) == 24:
            new_question_ind = 0
            for i in range(0, 24):
                if i in skip:
                    continue
                replace[str(new_question_ind)] = questions[str(i)]
                new_question_ind += 1
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


with open("quadrants.json") as f:
    data = json.load(f)

list_of_quad_data = []
for files in os.listdir('.\\building_ref_data'):
    with open(files) as quad_f:
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

# Creating ref axis
ref_axis = [i for i in range(1, 21)]

# Creating question axis
question_axis = [i for i in range(0, 5)]

# Users will be the axis
within_user = [[] for user_name in usernames]
within_builds = [[] for user_name in usernames]

# ref/question index will be axis
within_stimuli = [[] for i in range(0, 20)]
within_refs = [[] for i in range(0, 20)]

populate_within_stimuli(usernames, within_stimuli)
populate_within_refs(usernames, within_refs)
populate_within_user(usernames, within_user)
populate_within_builds(usernames, within_builds)
convert_to_pairwise_matrices(within_stimuli)
convert_to_pairwise_matrices(within_refs)
convert_to_pairwise_matrices(within_user)
convert_to_pairwise_matrices(within_builds)
convert_all_to_heatmap(within_stimuli, question_axis)
convert_all_to_heatmap(within_refs, ref_axis)
convert_all_to_heatmap(within_user, user_axis)
convert_all_to_heatmap(within_builds, user_axis)
