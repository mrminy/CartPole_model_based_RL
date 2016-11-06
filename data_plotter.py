"""
This file is for showing different graphs for different data
"""
import pprint
import matplotlib.pyplot as plt
import numpy as np


def show_space_utilization(data):
    pass

def convert_reward_data(d):
    out = []
    for game in d:
        g = []
        for val in game:
            if val != 0 and len(g) < 1000:
                g.append(val)
        out.append(g)
    return out

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


data = np.load('timesteps.npy')
print(data)
conv_data = convert_reward_data(data)
print(conv_data)
conv_data = np.array(conv_data)
np.savetxt("timesteps.csv", conv_data, delimiter=",")
# pprint.PrettyPrinter(indent=4).pprint(get_min_max(data[:, 0]))
