"""
This file is for showing different graphs for different data
"""
import pprint
import matplotlib.pyplot as plt
import numpy as np


def show_space_utilization(data):
    pass


def get_min_max(states):
    print(len(states))
    s_max = np.array(states[0])
    s_min = np.array(states[0])
    for s in states:
        for i in range(len(s)):
            if s[i] > s_max[i]:
                s_max[i] = s[i]
            if s[i] < s_min[i]:
                s_min[i] = s[i]
    return s_max, s_min


data = np.load('lunarlander_data/random_agent/testing_data.npy')
pprint.PrettyPrinter(indent=4).pprint(get_min_max(data[:, 0]))
