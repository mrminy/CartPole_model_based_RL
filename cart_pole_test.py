#!/usr/bin/env python
"""Script to solve cartpole control problem using policy gradient with neural network
	function approximation
	Polcy used is softmax
	Value function is approximated using a multi layered pereceptron with tanh units
	Critic (value network) is updated using monte-carlo prediction
	Actor (softmax policy) is updated using TD(0) for Advantage estimation

	This actor-critic implementation is fetched from mohakbhardwaj's open source implementation on Openai gym.
	Link: https://gym.openai.com/evaluations/eval_KhmXmmgmSManWEtxZJoLeg
	The implementation is slightly modified to meet my functionality requirements.
	"""
import gym
import numpy as np
from my_actor_critic import ActorCriticLearner


def main():
    save_history = False
    w_rollouts = False
    w_game_gui = False
    n_pre_training_epochs = 250
    n_rollout_epochs = 50
    n_agents = 3  # Train n different agents
    full_state_action_history = []
    end_episode = []
    for i in range(n_agents):
        print("Starting game nr:", i)
        env = gym.make('CartPole-v0')
        # env.seed(1234)
        # np.random.seed(1234)
        if w_game_gui:
            env.monitor.start('./cartpole_0-pg-experiment', force=True)
        # Learning Parameters
        max_episodes = 1000
        episodes_before_update = 2
        discount = 0.85
        ac_learner = ActorCriticLearner(env, max_episodes, episodes_before_update, discount, n_pre_training_epochs,
                                        n_rollout_epochs, logger=True, w_rollouts=w_rollouts)
        full_state_action_history.append(ac_learner.learn(200, 195))
        end_episode.append(ac_learner.last_episode)
        env.monitor.close()
    print("Done simulating...")
    print(end_episode)
    if save_history:
        np_state_history = []
        print("Converting and saving...")
        for i in range(len(full_state_action_history)):
            for j in range(len(full_state_action_history[i])):
                np_state_history.append(full_state_action_history[i][j])
        np_state_history = np.array(np_state_history)
        print(np_state_history.shape)
        np.save("cartpole_data/testing.npy", full_state_action_history)
        print("Done saving...")


if __name__ == "__main__":
    main()
