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
    w_game_gui = False
    gym_name = 'CartPole-v0'
    action_uncertainty = 0.0  # 4/10 when 0.3 solved. 0/10 when 0.4
    n_pre_training_epochs = 200
    n_rollout_epochs = 0  # Disabled for now..
    n_agents = 10  # Train n different agents
    learning_rate = 0.01
    pre_training_learning_rate = 0.001
    full_state_action_history = []
    end_episode = []
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
        ac_learner = ActorCriticLearner(env, max_episodes, episodes_before_update, discount, n_pre_training_epochs,
                                        n_rollout_epochs, action_uncertainty=action_uncertainty, logger=True,
                                        transition_model_restore_path='new_transition_model/random_agent_10000_2/transition_model.ckpt')
        full_state_action_history.append(ac_learner.learn(200, 195, learning_rate=learning_rate,
                                                          imagination_learning_rate=pre_training_learning_rate))
        end_episode.append(ac_learner.last_episode)
        env.monitor.close()
    print("Done simulating...")
    print(end_episode)
    print(np.mean(end_episode))
    print(np.std(end_episode))
    print(np.min(end_episode))


if __name__ == "__main__":
    main()
