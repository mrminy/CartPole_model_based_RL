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
    w_game_gui = True
    gym_name = 'CartPole-v0'
    action_uncertainty = 0.0  # 4/10 when 0.3 solved. 0/10 when 0.4
    n_pre_training_episodes = 200
    n_rollout_epochs = 0  # Disabled for now..
    n_agents = 1  # Train n different agents
    learning_rate = 0.01
    pre_training_learning_rate = 0.001
    full_state_action_history = []
    end_episode = []
    sum_steps = []
    step_history = []
    sum_rewards = []
    reward_history = []
    for i in range(n_agents):
        print("Starting game nr:", i)
        env = gym.make(gym_name)
        # env.seed(1234)
        # np.random.seed(1234)
        if w_game_gui:
            env.monitor.start('./' + gym_name + '-pg-experiment', force=True)
        # Learning Parameters
        max_episodes = 1000
        episodes_before_update = 2
        discount = 0.85
        ac_learner = ActorCriticLearner(env, max_episodes, episodes_before_update, discount, n_pre_training_episodes,
                                        n_rollout_epochs, action_uncertainty=action_uncertainty, logger=True)
        full_state_action_history.append(ac_learner.learn(env.spec.timestep_limit, env.spec.reward_threshold,
                                                          learning_rate=learning_rate,
                                                          imagination_learning_rate=pre_training_learning_rate))
        sum_steps.append(sum(env.monitor.stats_recorder.episode_lengths))
        step_history.append(env.monitor.stats_recorder.episode_lengths)
        sum_rewards.append(sum(env.monitor.stats_recorder.episode_rewards))
        reward_history.append(env.monitor.stats_recorder.episode_rewards)
        end_episode.append(ac_learner.last_episode)
        env.monitor.close()
    print("Done simulating...")
    print("Sum steps:", sum_steps)
    print("Sum rewards:", sum_rewards)
    print("End episodes:", end_episode)
    print("Mean:", np.mean(end_episode))
    print("Std:", np.std(end_episode))
    print("Best:", np.min(end_episode))

    np.savetxt("timesteps.csv", np.asarray(step_history), delimiter=",")
    np.savetxt("rewards.csv", np.asarray(reward_history), delimiter=",")



if __name__ == "__main__":
    main()
