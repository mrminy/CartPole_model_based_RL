"""
Script to solve cartpole control problem using policy gradient with neural network function approximation
Polcy used is softmax Value function is approximated using a multi layered pereceptron with tanh units
Critic (value network) is updated using monte-carlo prediction Actor (softmax policy) is updated using TD(0)
for Advantage estimation

The actor-critic implementation used is a modified version from mohakbhardwaj's open source actor-critic implementation
from OpenAI gym.
Link: https://gym.openai.com/evaluations/eval_KhmXmmgmSManWEtxZJoLeg
This implementation is slightly modified to be able to switch to model-based version.
"""
import gym
import numpy as np
from actor_critic import ActorCriticLearner


def save_data(step_history, reward_history, end_episode):
    """
    Saves experience to files
    :param step_history: number of time steps for each episode
    :param reward_history: rewards for each episode
    :param end_episode: number of episodes to used to solve the environment
    """
    step_history = np.array(step_history)
    reward_history = np.array(reward_history)

    print(reward_history)

    np.savetxt("end_episode.csv", np.asarray(end_episode), delimiter=",")
    np.save('timesteps.npy', step_history)
    np.save('rewards.npy', reward_history)


def main():
    w_game_gui = True
    gym_name = 'CartPole-v0'
    action_uncertainty = 0.0  # This can be altered to make CartPole nondeterministic
    n_pre_training_episodes = 200  # Number of pre-training episodes to be done with the dynamics models
    n_agents = 1  # Train n different agents
    learning_rate = 0.01  # Learning rate in real environment
    pre_training_learning_rate = 0.001  # Learning rate in model-based simulation
    full_state_action_history = []
    end_episode = []
    sum_steps = []
    step_history = []
    sum_rewards = []
    reward_history = []
    for i in range(n_agents):
        print("Starting game nr:", i)
        env = gym.make(gym_name)
        if w_game_gui:
            env.monitor.start('./' + gym_name + '-pg-experiment', force=True)

        # Learning Parameters
        max_episodes = 1000
        episodes_before_update = 2
        discount = 0.85
        ac_learner = ActorCriticLearner(env, max_episodes, episodes_before_update, discount, n_pre_training_episodes,
                                        action_uncertainty=action_uncertainty, logger=True)

        # Train the actor-critic
        full_state_action_history.append(ac_learner.learn(env.spec.timestep_limit, env.spec.reward_threshold,
                                                          learning_rate=learning_rate,
                                                          imagination_learning_rate=pre_training_learning_rate))
        sum_steps.append(sum(env.monitor.stats_recorder.episode_lengths))
        sum_rewards.append(sum(env.monitor.stats_recorder.episode_rewards))
        step_history.append(env.monitor.stats_recorder.episode_lengths)
        reward_history.append(env.monitor.stats_recorder.episode_rewards)
        end_episode.append(ac_learner.last_episode)
        env.monitor.close()
        save_data(step_history, reward_history, end_episode)
    print("Done simulating...")
    print("Sum steps:", sum_steps)
    print("Sum rewards:", sum_rewards)
    print("End episodes:", end_episode)
    print("Mean:", np.mean(end_episode))
    print("Std:", np.std(end_episode))
    print("Best:", np.min(end_episode))

    save_data(step_history, reward_history, end_episode)


if __name__ == "__main__":
    main()
