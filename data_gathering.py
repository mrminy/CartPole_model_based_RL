# from my_actor_critic import ActorCriticLearner
import gym
import numpy as np
import time
from my_actor_critic import ActorCriticLearner


def gather_data_random_agent(n_games, max_time_steps, env, save_path="lunarlander_data/training_data.npy", save_data=True):
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


def gather_data_actor_critic(n_agents, max_episodes, max_time_steps, env, save_path="cartpole_data/training_data.npy",
                             save_data=True):
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
    env = gym.make('MsPacman-ram-v0')

    # Random agent sampling
    d = gather_data_random_agent(1000, 1000, env,
                                 save_path="mspacman_data_done/random_agent/training_data.npy",
                                 save_data=True)

    # d = gather_data_actor_critic(10, 1000, 1000, env,
    #                              save_path="lunarlander_data_done/actor_critic/training_data.npy",
    #                              save_data=True)

    # Actor critic sampling
    # d = gather_data_actor_critic(30, 1000, 200, env, save_path="cartpole_data/actor_critic_testing_data.npy",
    #                              save_data=True)

    print("Number of steps:", len(d))
    print("Done:", time.time() - start_time)
