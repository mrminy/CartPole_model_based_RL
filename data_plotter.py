"""
Not an important script. Used for converting numpy arrays into nice csv-files...
"""
import numpy as np


def convert_reward_data(d):
    out = []
    for game in d:
        g = np.zeros(1200)
        i = 0
        for val in game:
            if val != 0:
                g[i] = val
                i += 1
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

if __name__ == '__main__':
    data = np.load('lunar_rewards.npy')
    print(data)
    conv_data = convert_reward_data(data)
    print(conv_data)
    conv_data = np.array(conv_data)
    np.savetxt("lunar_rewards.csv", conv_data, delimiter=",")

    data = np.load('lunar_timesteps.npy')
    print(data)
    conv_data = convert_reward_data(data)
    print(conv_data)
    conv_data = np.array(conv_data)
    np.savetxt("lunar_timesteps.csv", conv_data, delimiter=",")
