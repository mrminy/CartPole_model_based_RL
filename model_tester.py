import numpy as np
import gym
from model_based_learner import TF_Transition_model
import matplotlib.pyplot as plt

results = []
env = gym.make('CartPole-v0')
env.reset()
last_step = 0
tr_model = TF_Transition_model(env)
tr_model.restore_model(restore_path='new_transition_model/actor_critic/transition_model.ckpt')

state = env.reset()
im_state = state

real_states = [state]
real_rewards = [1.0]
im_states = [im_state]
im_rewards = [1.0]

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

print(real_states)
print(im_states)
print(real_rewards)
print(im_rewards)

real_states = np.array(real_states)
im_states = np.array(im_states)

test_index = [0]
print(test_index)
print(np.take(real_states, test_index, 1).reshape(len(real_states)))
print(np.take(im_states, test_index, 1).reshape(len(im_states)))
f_0 = plt.figure()
ax0 = f_0.add_subplot(111)
ax0.plot(np.take(real_states, test_index, 1))
ax0.plot(np.take(im_states, test_index, 1))

test_index = [1]
print(test_index)
print(np.take(real_states, test_index, 1).reshape(len(real_states)))
print(np.take(im_states, test_index, 1).reshape(len(im_states)))
f_1 = plt.figure()
ax1 = f_1.add_subplot(111)
ax1.set_title('Index ' + str(test_index[0]))
ax1.plot(np.take(real_states, test_index, 1))
ax1.plot(np.take(im_states, test_index, 1))

test_index = [2]
print(test_index)
print(np.take(real_states, test_index, 1).reshape(len(real_states)))
print(np.take(im_states, test_index, 1).reshape(len(im_states)))
f_2 = plt.figure()
ax2 = f_2.add_subplot(111)
ax2.set_title('Index ' + str(test_index[0]))
ax2.plot(np.take(real_states, test_index, 1))
ax2.plot(np.take(im_states, test_index, 1))

test_index = [3]
print(test_index)
print(np.take(real_states, test_index, 1).reshape(len(real_states)))
print(np.take(im_states, test_index, 1).reshape(len(im_states)))
f_3 = plt.figure()
ax3 = f_3.add_subplot(111)
ax3.set_title('Index ' + str(test_index[0]))
ax3.plot(np.take(real_states, test_index, 1))
ax3.plot(np.take(im_states, test_index, 1))


plt.show()
