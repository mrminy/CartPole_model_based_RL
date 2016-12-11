import gym
import numpy as np
import time
from actor_critic import ActorCriticLearner


def gather_data_random_agent(env, max_time_steps, n_games=1000, save_path="lunarlander_data_done/training_data.npy",
                             save_data=True):
    """
    Gather data by following random action policy for n games
    :param env: the environment
    :param max_time_steps: maximum number of time steps per episode
    :param n_games: number of games to sample from
    :param save_path: save path
    :param save_data: save or not?
    :return: all experience collected
    """
    data = []
    for i in range(n_games):
        if i % 1000 == 0:
            print("Game nr:", i)
        curr_state = env.reset()
        for time in range(max_time_steps):
            action = np.random.choice(env.action_space.n)
            # Execute the action in the environment and observe reward
            next_state, reward, done, info = env.step(action)
            data.append(np.array([curr_state, action, next_state, reward, done]))
            curr_state = next_state
            if done:
                break

    data = np.array(data)
    if save_data:
        np.save(save_path, data)
        print("Done saving...")
    return data


def gather_data_actor_critic(n_agents, max_episodes, max_time_steps, env,
                             save_path="cartpole_data_done/actor_critic/training_data.npy", save_data=True):
    """
    Gathers data while learning an agent with actor-critic
    :param n_agents: n different agents to learn
    :param max_episodes: maximum number of episodes per agent
    :param max_time_steps: maximum time steps per episode
    :param env: the environment (CartPole)
    :param save_path: save path
    :param save_data: save or not?
    :return: all experience collected
    """
    data = []
    for i in range(n_agents):
        print("Gathering data from agent nr:", i)
        env.reset()
        actor_critic = ActorCriticLearner(env, max_episodes, 2, 0.85, n_pre_training_epochs=0, n_rollout_epochs=0,
                                          logger=False)
        d = actor_critic.learn(max_time_steps, 195)
        for d_sample in d:
            for single_samlple in d_sample:
                data.append(np.array(single_samlple))

    data = np.array(data)
    if save_data:
        np.save(save_path, data)
        print("Done saving...")
    return data


if __name__ == '__main__':
    start_time = time.time()
    env = gym.make('LunarLander-v2')

    # Random agent sampling from LunarLander
    d = gather_data_random_agent(env, env.spec.timestep_limit, 1000,
                                 save_path="lunarlander_data_done/random_agent/training_data.npy", save_data=True)

    # Actor critic sampling for cartpole
    # d = gather_data_actor_critic(30, 1000, 200, env, save_path="cartpole_data/actor_critic_testing_data.npy",
    #                              save_data=True)

    print("Number of steps:", len(d))
    print("Done:", time.time() - start_time)
