# Deep Q network
import gym
import numpy as np
import tensorflow as tf
import random
from model_based_learner import TF_Transition_model

# HYPERPARMETERS
H = 200
H2 = 200
batch_number = 500
gamma = 0.99
num_between_q_copies = 150
explore_decay = 0.9995  # Exploration decay per step
min_explore = 0.01
max_steps = 1000
reward_goal = 200
memory_size = 1000000
learning_rate = 0.008
w_game_gui = True

n_imagination_rollouts = 0

class DQN:
    def __init__(self, env):
        # Set up the environment
        self.env = env

        self.transition_model = TF_Transition_model(env)
        # self.transition_model.restore_model(
        #     restore_path='new_transition_model/random_agent_1000/transition_model.ckpt')

        self.graph = tf.Graph()
        self.all_assigns = None
        self.Q = None
        self.Q_ = None
        self.optimize = None
        self.init = None
        self.target_q = None
        self.states_ = None
        self.action_used = None

        self.D = []
        self.build()

    def build(self):
        # First Q Network
        w1 = tf.Variable(tf.random_uniform([self.env.observation_space.shape[0], H], -1.0, 1.0))
        b1 = tf.Variable(tf.random_uniform([H], -1.0, 1.0))

        w2 = tf.Variable(tf.random_uniform([H, H2], -1.0, 1.0))
        b2 = tf.Variable(tf.random_uniform([H2], -1.0, 1.0))

        w3 = tf.Variable(tf.random_uniform([H2, self.env.action_space.n], -1.0, 1.0))
        b3 = tf.Variable(tf.random_uniform([self.env.action_space.n], -1.0, 1.0))

        # Second Q Network
        w1_ = tf.Variable(tf.random_uniform([self.env.observation_space.shape[0], H], -1.0, 1.0))
        b1_ = tf.Variable(tf.random_uniform([H], -1.0, 1.0))

        w2_ = tf.Variable(tf.random_uniform([H, H2], -1.0, 1.0))
        b2_ = tf.Variable(tf.random_uniform([H2], -1.0, 1.0))

        w3_ = tf.Variable(tf.random_uniform([H2, self.env.action_space.n], -1.0, 1.0))
        b3_ = tf.Variable(tf.random_uniform([self.env.action_space.n], -1.0, 1.0))

        # Make assign functions for updating Q prime's weights
        w1_update = w1_.assign(w1)
        b1_update = b1_.assign(b1)
        w2_update = w2_.assign(w2)
        b2_update = b2_.assign(b2)
        w3_update = w3_.assign(w3)
        b3_update = b3_.assign(b3)

        self.all_assigns = [
            w1_update,
            w2_update,
            w3_update,
            b1_update,
            b2_update,
            b3_update]

        # build network
        self.states_ = tf.placeholder(tf.float32, [None, self.env.observation_space.shape[0]])
        h_1 = tf.nn.relu(tf.matmul(self.states_, w1) + b1)
        h_2 = tf.nn.relu(tf.matmul(h_1, w2) + b2)
        h_2 = tf.nn.dropout(h_2, .5)
        self.Q = tf.matmul(h_2, w3) + b3

        h_1_ = tf.nn.relu(tf.matmul(self.states_, w1_) + b1_)
        h_2_ = tf.nn.relu(tf.matmul(h_1_, w2_) + b2_)
        h_2_ = tf.nn.dropout(h_2_, .5)
        self.Q_ = tf.matmul(h_2_, w3_) + b3_

        self.action_used = tf.placeholder(tf.int32, [None], name="action_masks")
        action_masks = tf.one_hot(self.action_used, self.env.action_space.n)
        filtered_Q = tf.reduce_sum(tf.mul(self.Q, action_masks), reduction_indices=1)

        # Train Q
        self.target_q = tf.placeholder(tf.float32, [None, ])
        loss = tf.reduce_mean(tf.square(filtered_Q - self.target_q))
        self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        self.init = tf.initialize_all_variables()

    def train(self, max_episodes=1000, pre_learning_episodes=200, im_rollouts=True, logger=True):
        explore = 1.0  # Keep minimal exploration in real life
        reward_list = []
        recent_rewards = []
        past_actions = []
        episode_number = 0
        episode_reward = 0
        reward_sum = 0
        episode = 0
        last_reward_sum = 0

        if w_game_gui:
            self.env.monitor.start('./' + self.env.spec.id + '-pg-experiment', force=True)

        with tf.Session() as sess:
            sess.run(self.init)
            sess.run(self.all_assigns)

            # Pre-training with transition model
            self.pre_learn(sess, max_episodes=pre_learning_episodes, logger=logger)

            ticks = 0
            for episode in range(max_episodes):
                state = self.env.reset()
                start_state = state

                reward_sum = 0

                for step in range(max_steps):
                    ticks += 1

                    if episode % 10 == 0:
                        q, qp = sess.run([self.Q, self.Q_], feed_dict={self.states_: np.array([state])})
                        # print "Q:{}, Q_ {}".format(q[0], qp[0])
                        # env.render()

                    if explore > random.random():
                        action = self.env.action_space.sample()
                    else:
                        q = sess.run(self.Q, feed_dict={self.states_: np.array([state])})[0]
                        action = np.argmax(q)
                    explore = max(explore * explore_decay, min_explore)

                    new_state, reward, done, _ = self.env.step(action)
                    reward_sum += reward

                    self.D.append([state, action, reward, new_state, done])
                    if len(self.D) > memory_size:
                        self.D.pop(0)

                    state = new_state

                    if done:
                        break

                    if last_reward_sum < reward_goal:
                        # Training a Batch
                        samples = random.sample(self.D, min(batch_number, len(self.D)))

                        # calculate all next Q's together
                        new_states = [x[3] for x in samples]
                        all_q = sess.run(self.Q_, feed_dict={self.states_: new_states})

                        y_ = []
                        state_samples = []
                        actions = []
                        terminalcount = 0
                        for ind, i_sample in enumerate(samples):
                            state_mem, curr_action, reward, new_state, done = i_sample
                            if done:
                                y_.append(reward)
                                terminalcount += 1
                            else:
                                this_q = all_q[ind]
                                maxq = max(this_q)
                                y_.append(reward + (gamma * maxq))

                            state_samples.append(state_mem)

                            actions.append(curr_action)
                        sess.run([self.optimize],
                                 feed_dict={self.states_: state_samples, self.target_q: y_, self.action_used: actions})
                        if ticks % num_between_q_copies == 0:
                            sess.run(self.all_assigns)

                # Do imagination rollouts
                for r in range(n_imagination_rollouts):
                    if im_rollouts and len(self.D) < memory_size:
                        # Choose random action for imagination rollouts
                        im_done = False
                        im_state = start_state
                        while not im_done:
                            # Only explore in simulation
                            im_action = self.env.action_space.sample()
                            im_new_state, im_reward, im_done, _ = self.transition_model.predict(im_state,
                                                                                                im_action)
                            self.D.append([im_state, im_action, im_reward, im_new_state, im_done])
                            if len(self.D) > memory_size:
                                self.D.pop(0)
                            im_state = im_new_state

                last_reward_sum = reward_sum
                reward_list.append(reward_sum)
                recent_rewards.append(reward_sum)
                if len(recent_rewards) > 100:
                    recent_rewards.pop(0)

                mean = 0.0
                if len(recent_rewards) > 0:
                    mean = sum(recent_rewards) / float(len(recent_rewards))
                if logger:
                    print(
                        'Reward for episode %d is %d. Explore is %.4f. Avg is %.3f. ER size is %d' % (
                            episode, reward_sum, explore, mean, len(self.D)))

                if mean >= reward_goal and len(recent_rewards) >= 100:
                    break

            if logger:
                print("Steps:", ticks)

        self.env.monitor.close()
        if logger:
            print("Last episode:", episode)
        return episode

    def pre_learn(self, sess, max_episodes=200, logger=False):
        explore = 1.0
        reward_list = []
        recent_rewards = []
        # past_actions = []
        # episode_number = 0
        # episode_reward = 0

        ticks = 0
        for episode in range(max_episodes):
            state = self.env.reset()

            reward_sum = 0

            for step in range(max_steps):
                ticks += 1

                if episode % 10 == 0:
                    q, qp = sess.run([self.Q, self.Q_], feed_dict={self.states_: np.array([state])})
                    # print "Q:{}, Q_ {}".format(q[0], qp[0])
                    # env.render()

                if explore > random.random() or True:
                    action = self.env.action_space.sample()
                else:
                    q = sess.run(self.Q, feed_dict={self.states_: np.array([state])})[0]
                    action = np.argmax(q)
                # action = self.env.action_space.sample()
                explore = max(explore * explore_decay, min_explore)

                new_state, reward, done, _ = self.transition_model.predict(state, action)  # self.env.step(action)
                reward_sum += reward

                self.D.append([state, action, reward, new_state, done])
                if len(self.D) > memory_size:
                    self.D.pop(0)

                state = new_state

                if done:
                    break

                # Training a Batch
                samples = random.sample(self.D, min(batch_number, len(self.D)))

                # calculate all next Q's together
                new_states = [x[3] for x in samples]
                all_q = sess.run(self.Q_, feed_dict={self.states_: new_states})

                y_ = []
                state_samples = []
                actions = []
                terminalcount = 0
                for ind, i_sample in enumerate(samples):
                    state_mem, curr_action, reward, new_state, done = i_sample
                    if done:
                        y_.append(reward)
                        terminalcount += 1
                    else:
                        this_q = all_q[ind]
                        maxq = max(this_q)
                        y_.append(reward + (gamma * maxq))

                    state_samples.append(state_mem)

                    actions.append(curr_action)
                sess.run([self.optimize],
                         feed_dict={self.states_: state_samples, self.target_q: y_, self.action_used: actions})
                if ticks % num_between_q_copies == 0:
                    sess.run(self.all_assigns)

            reward_list.append(reward_sum)
            recent_rewards.append(reward_sum)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)

            mean = 0.0
            if len(recent_rewards) > 0:
                mean = sum(recent_rewards) / float(len(recent_rewards))
            if logger:
                print('Pre-learning --> Reward for episode %d is %d. Explore is %.4f. Avg is %.3f' % (
                    episode, reward_sum, explore, mean))

            if mean >= reward_goal and len(recent_rewards) >= 100:
                break

                # Clear ER for real and correct episodes
                # self.D = []
                # print("Pre-learning steps:", ticks)


if __name__ == '__main__':
    results = []
    for i in range(1):
        print("Agent nr:", i)
        dqn = DQN(gym.make('LunarLander-v2'))
        results.append(
            dqn.train(max_episodes=10000, pre_learning_episodes=0, im_rollouts=False, logger=True))
    print(results)
    results = np.array(results)
    print("Avg:", np.mean(results))
    print("Std:", np.std(results))
