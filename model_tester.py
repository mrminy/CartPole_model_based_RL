import numpy as np
import gym
import matplotlib.pyplot as plt

from model_based_learner import TF_Transition_model
# from model_based_learner_lunar_2 import TF_Transition_model

results = []
env = gym.make('CartPole-v0')
env.reset()

# training_data = np.load('lunarlander_data_done/random_agent/training_data.npy')
# test_data_r = np.load('lunarlander_data_done/random_agent/testing_data.npy')
tr_model = TF_Transition_model(env, history_sampling_rate=1, w_init_limit=(-0.2, 0.2), display_step=1)
# tr_model.train(training_epochs=200, learning_rate=0.0007, training_data=training_data, batch_size=512, test_data_r=test_data_r,
#                test_data_ac=None, save=False, save_path="transition_model.ckpt", max_training_data=50000)
tr_model.restore_model(restore_path='transition_model_saves/cartpole/1k/transition_model_1k.ckpt')
last_step = 0
while last_step < 50:
    last_step = 0
    state = env.reset()
    im_state = state

    real_states = [state]
    real_rewards = [0.0]
    im_states = [im_state]
    im_rewards = [0.0]

    for time_step in range(200):
        exec_action = np.random.choice(env.action_space.n)
        next_state, reward, done, info = env.step(exec_action)
        im_next_state, im_reward, im_done, _ = tr_model.predict(im_state, exec_action)

        real_states.append(next_state)
        real_rewards.append(reward)
        im_states.append(im_next_state)
        im_rewards.append(im_reward)

        state = next_state
        im_state = im_next_state
        last_step += 1
        if done:
            break

# print(real_states)
# print(im_states)
# print(real_rewards)
# print(im_rewards)

print("Number of steps done:", last_step)

real_states = np.array(real_states)
im_states = np.array(im_states)

# Plot all of the state inputs for CartPole
for index in range(4):
    test_index = [index]
    print(test_index)
    print(np.take(real_states, test_index, 1).reshape(len(real_states)))
    print(np.take(im_states, test_index, 1).reshape(len(im_states)))
    f_0 = plt.figure()
    ax0 = f_0.add_subplot(111)
    ax0.set_title('Index ' + str(test_index[0] + 1))
    ax0.plot(np.take(real_states, test_index, 1))
    ax0.plot(np.take(im_states, test_index, 1))

plt.show()
