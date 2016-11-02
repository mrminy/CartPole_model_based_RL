import numpy as np


def get_n_action(n):
    if n <= 2:
        return 1
    return n


def reward_function(state):
    # True reward function for CartPole
    theta_threshold_radians = 12 * 2 * np.math.pi / 360
    x_threshold = 2.4
    done = state[0] < -x_threshold or state[0] > x_threshold or state[2] < -theta_threshold_radians or state[
                                                                                                           2] > theta_threshold_radians
    done = bool(done)

    # Give reward of 1 if game is not over
    reward = 1.0
    if done:
        reward = 0.0
    return reward, done
