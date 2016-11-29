'''
Created on Jun 26, 2016
@author: Davide Nitti
https://github.com/davidenitti/ML/tree/master/RL/gymeval
'''

import argparse
import matplotlib.pyplot as plt
import time

import numpy as np
import gym
import agents_lunar_2 as agents


def save_data(step_history, reward_history):
    step_history = np.array(step_history)
    reward_history = np.array(reward_history)

    print(reward_history)

    # np.savetxt("end_episode.csv", np.asarray(end_episode), delimiter=",")
    np.save('pacman_timesteps.npy', step_history)
    np.save('pacman_rewards.npy', reward_history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', nargs="?", default="MsPacman-ram-v0")
    parser.add_argument('--seed', nargs="?", default=None)
    parser.add_argument('--episodes', nargs="?", default=1000)
    n_agents = 10
    args = parser.parse_args()
    step_history = []
    reward_history = []
    for i in range(n_agents):
        print(args)
        numepisodes = int(args.episodes)
        if args.seed is not None:
            seed = int(args.seed)
        else:
            seed = None
        nameenv = args.target

        env = gym.make(nameenv)
        reward_threshold = env.spec.reward_threshold

        if seed is not None:
            env.seed(seed)
            np.random.seed(seed)

        resultsdir = './' + nameenv

        env.monitor.start(resultsdir, force=True)
        print(env.observation_space, env.action_space, env.spec.timestep_limit, env.reward_range, gym.envs.registry.spec(
            nameenv).trials)
        if nameenv == 'Acrobot-v0':
            env.reward_range = (-1., 0.)

        if nameenv == 'LunarLander-v2':
            params = {
                "memsize": 150000,
                "scalereward": 1.,
                "probupdate": .25,
                "lambda": 0.15,
                "past": 0,
                "eps": 0.45,  # Epsilon in epsilon greedy policies
                "decay": 0.993,  # Epsilon decay in epsilon greedy policies
                "initial_learnrate": 0.012,
                "decay_learnrate": 0.997,
                "discount": 0.99,
                "batch_size": 75,
                "hiddenlayers": [300],
                "regularization": [0.00001, 0.00000001],
                "momentum": 0.05,
                "file": None,
                "seed": seed}
        else:
            params = {
                "memsize": 150000,
                "scalereward": 1.,
                "probupdate": .25,
                "lambda": 0.15,
                "past": 0,
                "eps": 0.45,  # Epsilon in epsilon greedy policies
                "decay": 0.993,  # Epsilon decay in epsilon greedy policies
                "initial_learnrate": 0.012,
                "decay_learnrate": .997,
                "discount": 0.99,
                "batch_size": 75,
                "hiddenlayers": [300],
                "regularization": [0.00001, 0.00000001],
                "momentum": 0.05,
                "file": None,
                "seed": seed}
        agent = agents.deepQAgent(env, env.observation_space, env.action_space, env.reward_range, **params)
        num_steps = env.spec.timestep_limit
        avg = 0.
        oldavg = 0.

        pre_training_episodes = 0
        agents.do_imagination_rollouts(agent, env, pre_training_episodes)
        print("DONE pre-training", pre_training_episodes, "episodes")

        plt.ion()
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[1].set_xlim(-1, 1)
        ax[1].set_ylim(-1, 1)
        ax[1].autoscale(False)

        totrewlist = []
        totrewavglist = []
        costlist = []
        sum_rewards = []
        showevery = 10
        for episode in range(numepisodes):
            if episode % showevery == 0:
                render = True
                eps = None
                print('episode', episode, 'l rate', agent.getlearnrate())
                oldavg = avg
            else:
                render = False
                eps = episode
            startt = time.time()
            total_rew, steps, cost, listob, listact = agents.do_rollout(agent, env, eps, render=render)

            if episode == 0:
                avg = total_rew
            if episode % 50 == 0:
                print(agent.config)
            if episode % 1 == 0:
                listob = np.array(listob)
                listact = np.array(listact)
                allactionsparse = np.zeros((listact.shape[0], agent.n_out))
                allactionsparse[np.arange(listact.shape[0]), listact] = 1.

                inc = max(0.06, 1. / (episode + 1.) ** 0.6)
                avg = avg * (1 - inc) + inc * total_rew
                totrewlist.append(total_rew / agent.config['scalereward'])
                totrewavglist.append(avg / agent.config['scalereward'])
                costlist.append(cost / steps)
                plotlast = 200 + episode % 50
                ax[0].clear()
                ax[0].set_xlim(max(-1, int(len(totrewavglist) / 50 * 50 - 150)), int(len(totrewavglist) / 50 * 50 + 50))
                ax[0].set_ylim(min(totrewlist[max(0, int(len(totrewavglist) / 50 * 50 - 150)):]) - 5,
                               max(totrewlist[max(0, int(len(totrewavglist) / 50 * 50 - 150)):]) + 5)
                ax[0].plot([0, len(totrewavglist) + 100], [reward_threshold, reward_threshold], color='green')

                ax[0].plot(range(len(totrewlist)), totrewlist, color='red')
                ax[0].plot(range(len(totrewavglist)), totrewavglist, color='blue')
                ax[0].scatter([showevery * ff for ff in range(len(totrewlist[::showevery]))], totrewlist[::showevery],
                              color='black')
                ax[0].plot([max(0, len(totrewavglist) - 1 - 100), len(totrewavglist) - 1],
                           [np.mean(np.array(totrewlist[-100:])), np.mean(np.array(totrewlist[-100:]))], color='black')
                plt.draw()
                plt.pause(.01)
            print(render, 'time', (time.time() - startt) / steps * 100., 'steps', steps, 'total reward', total_rew / \
                  agent.config[
                      'scalereward'], 'avg', avg / \
                  agent.config['scalereward'], cost / steps, 'eps', agent.epsilon(eps), len(agent.memory))
        step_history.append(env.monitor.stats_recorder.episode_lengths)
        reward_history.append(env.monitor.stats_recorder.episode_rewards)
        save_data(step_history, reward_history)

        # fig.savefig('last.png')
        # env.monitor.close()
        print(agent.config)
        # gym.upload(resultsdir, api_key='YOURAPI')
        env.monitor.close()
    step_history = np.array(step_history)
    reward_history = np.array(reward_history)
    save_data(step_history, reward_history)
