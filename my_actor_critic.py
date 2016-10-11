"""
This actor-critic agent is a modification from mohakbhardwaj's version. This is now a model-based implementation instead
of a model-free. The transition model is pre-learned by gathered data and supervised learning. See model_based_learner.py
"""
import random

import numpy as np
import tensorflow as tf
import tflearn

import model_based_learner

replay_states = []
replay_actions = []
replay_rewards = []
replay_next_states = []
replay_return_from_states = []

# Load reward model once
print("Generating reward model...")
reward_model = model_based_learner.create_reward_model()
print("Loading reward model...")
reward_model.load('reward_model/cart_pole_reward_model.tflearn')

# Load transition_model once
print("Loading transition model...")
transition_model = model_based_learner.TF_Transition_model()
# transition_model.load('transition_model/cart_pole_transition_model.tflearn', weights_only=True)

class Actor:
    def __init__(self, env, discount=0.90, learning_rate=0.01, w_rollouts=True):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.action_space_n = self.action_space.n
        # Learning parameters
        self.learning_rate = learning_rate
        self.discount = discount
        self.max_reward_for_game = -99999999.99
        # Declare tf graph
        self.graph = tf.Graph()

        self.w_rollouts = w_rollouts
        self.transition_model = transition_model
        self.reward_model = reward_model

        # Build the graph when instantiated
        with self.graph.as_default():
            tf.set_random_seed(1234)
            self.weights = tf.Variable(tf.random_normal([len(self.observation_space.high), self.action_space_n]))
            self.biases = tf.Variable(tf.random_normal([self.action_space_n]))

            # Neural Network inputs
            # The types of inputs possible include: state, advantage, action(to return probability of executing that action)
            self.x = tf.placeholder("float", [None, len(self.observation_space.high)])  # State input
            self.y = tf.placeholder("float")  # Advantage input
            self.action_input = tf.placeholder("float", [None,
                                                         self.action_space_n])  # Input action to return the probability associated with that action

            # Current policy is a simple softmax policy since actions are discrete in this environment
            self.policy = self.softmax_policy(self.x, self.weights, self.biases)  # Softmax policy
            # The following are derived directly from the formula for gradient of policy
            self.log_action_probability = tf.reduce_sum(self.action_input * tf.log(self.policy))
            self.loss = -self.log_action_probability * self.y  # Loss is score function times advantage
            # Use Adam Optimizer to optimize
            # [TODO: Add Trust Region Policy Optimization(TRPO)]
            self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            # Initializing all variables
            self.init = tf.initialize_all_variables()
            print("Policy Graph Constructed")

        # Declare a TF session and initialize it
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def rollout_policy(self, timeSteps):
        """Rollout policy for one episode, update the replay memory and return total reward"""
        total_reward = 0
        curr_state = self.env.reset()
        # Initialize lists in order to store episode data
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        episode_return_from_states = []

        for time in range(timeSteps):
            # Choose action based on current policy
            action = self.choose_action(curr_state)
            # Execute the action in the environment and observe reward
            next_state, reward, done, info = self.env.step(action)
            # Update the total reward
            total_reward += reward
            if done or time >= self.env.spec.timestep_limit:
                # print "Episode {} ended at step {} with total reward {}".format(episodeNumber, time, total_reward)
                break

            # Add state, action, reward transitions to containers for episode data
            # [TODO: Store discounted return instead of just return to test]
            curr_state_l = curr_state.tolist()
            next_state_l = next_state.tolist()
            if curr_state_l not in episode_states:
                episode_states.append(curr_state_l)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_next_states.append(next_state_l)
                episode_return_from_states.append(reward)
                for i in range(len(episode_return_from_states) - 1):
                    # Here multiply the reward by discount factor raised to the power len(episode_return_from_states)-1-i
                    # episode_return_from_states[i] += reward
                    episode_return_from_states[i] += pow(self.discount,
                                                         len(episode_return_from_states) - 1 - i) * reward
            else:
                # Iterate through the replay memory and update the final return for all states, i.e don't add the
                # state if it is already there but update reward for other states
                for i in range(len(episode_return_from_states)):
                    episode_return_from_states[i] += pow(self.discount, len(episode_return_from_states) - i) * reward

            curr_state = next_state
        if total_reward > self.max_reward_for_game:
            self.max_reward_for_game = total_reward

        if self.w_rollouts:
            # Make imagination rollouts from model to gain even more experience from a random action policy
            rollout_episode_states, rollout_episode_actions, rollout_episode_rewards, rollout_episode_next_states, rollout_episode_return_from_states = self.perform_imagination_rollouts(
                episode_states, episode_next_states[-1])
            episode_states = rollout_episode_states
            episode_actions = rollout_episode_actions
            episode_rewards = rollout_episode_rewards
            episode_next_states = rollout_episode_next_states
            episode_return_from_states = rollout_episode_return_from_states

        # Update the global replay memory
        self.update_memory(episode_states, episode_actions, episode_rewards, episode_next_states,
                           episode_return_from_states)
        return episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states, total_reward

    def perform_imagination_rollouts(self, real_life_experience, last_episode):
        # st0 = episode_states[n], st = episode_states[n+1], a = random_action, s' = model(st0, st, a), r = r_model(s')
        # Make into a function
        rollout_episode_states = []
        rollout_episode_actions = []
        rollout_episode_rewards = []
        rollout_episode_next_states = []
        rollout_episode_return_from_states = []
        for n in range(len(real_life_experience)):
            st0 = real_life_experience[n][:]
            if n + 1 < len(real_life_experience):
                st = real_life_experience[n + 1][:]
            else:
                st = last_episode[:]
            a = np.random.choice(self.env.action_space.n)
            state_representation = model_based_learner.generate_state_representation(st0, st, a)
            s_dash = self.transition_model.predict(st0, st, a)
            r_dash = self.reward_model.predict([state_representation])[0][0]

            curr_state_l = st
            next_state_l = s_dash

            if curr_state_l not in rollout_episode_states:
                rollout_episode_states.append(curr_state_l)
                rollout_episode_actions.append(a)
                rollout_episode_rewards.append(r_dash)
                rollout_episode_next_states.append(next_state_l)
                rollout_episode_return_from_states.append(r_dash)
                for i in range(len(rollout_episode_return_from_states) - 1):
                    # Here multiply the reward by discount factor raised to the power len(episode_return_from_states)-1-i
                    # episode_return_from_states[i] += reward
                    rollout_episode_return_from_states[i] += pow(self.discount, len(rollout_episode_return_from_states) - 1 - i) * r_dash
            else:
                # Iterate through the replay memory and update the final return for all states, i.e don't add the
                # state if it is already there but update reward for other states
                for i in range(len(rollout_episode_return_from_states)):
                    rollout_episode_return_from_states[i] += pow(self.discount,
                                                                 len(rollout_episode_return_from_states) - i) * r_dash

        # # Update the global replay memory from rollout experience
        # self.update_memory(rollout_episode_states, rollout_episode_actions, rollout_episode_rewards,
        #                    rollout_episode_next_states, rollout_episode_return_from_states)
        return rollout_episode_states, rollout_episode_actions, rollout_episode_rewards, rollout_episode_next_states, rollout_episode_return_from_states

    def update_policy(self, advantage_vectors):
        """Updates the policy weights by running gradient descent on one state at a time"""
        # [TODO: Try out batch gradient descent in this case as well]
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states

        for i in range(len(replay_states)):

            states = replay_states[i]
            actions = replay_actions[i]
            advantage_vector = advantage_vectors[i]
            for j in range(len(states)):
                action = self.to_action_input(actions[j])

                state = np.asarray(states[j])
                state = state.reshape(1, len(self.observation_space.high))

                _, error_value = self.sess.run([self.optim, self.loss],
                                               feed_dict={self.x: state, self.action_input: action,
                                                          self.y: advantage_vector[j]})

    def softmax_policy(self, state, weights, biases):
        """Defines softmax policy for tf graph"""
        policy = tf.nn.softmax(tf.matmul(state, weights) + biases)
        return policy

    def choose_action(self, state):
        """Chooses action from the crrent policy and weights"""
        if random.random() < 0.0:
            # Try with some epsilon greedy exploration together with the softmax action policy
            action = np.random.choice(self.env.action_space.n)
        else:
            state = np.asarray(state)
            state = state.reshape(1, len(self.observation_space.high))
            softmax_out = self.sess.run(self.policy, feed_dict={self.x: state})
            action = np.random.choice([0, 1], 1, replace=True, p=softmax_out[0])[0]  # Sample action from prob density
        return action

    def update_memory(self, episode_states, episode_actions, episode_rewards, episode_next_states,
                      episode_return_from_states):
        """Updates the global replay memory"""
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
        # Using first visit Monte Carlo so total return from a state is calculated from first time it is visited

        replay_states.append(episode_states)
        replay_actions.append(episode_actions)
        replay_rewards.append(episode_rewards)
        replay_next_states.append(episode_next_states)
        replay_return_from_states.append(episode_return_from_states)

    def reset_memory(self):
        """Resets the global replay memory"""
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
        del replay_states[:], replay_actions[:], replay_rewards[:], replay_next_states[:], replay_return_from_states[:]

    def to_action_input(self, action):
        """Utility function to convert action to a format suitable for the neura networ input"""
        action_input = [0] * self.action_space_n
        # print "Action going in: ", action
        action_input[action] = 1
        action_input = np.asarray(action_input)
        action_input = action_input.reshape(1, self.action_space_n)
        return action_input


class Critic:
    """Defines the critic network and functions to evaluate the current policy using Monte Carlo"""

    def __init__(self, env, discount=0.90, learning_rate=0.008):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.action_space_n = self.action_space.n
        self.n_input = len(self.observation_space.high)
        self.n_hidden_1 = 20
        # Learning Parameters
        self.learning_rate = learning_rate
        self.discount = discount
        self.num_epochs = 20  # 20 works
        self.batch_size = 32  # 170 works
        self.graph = tf.Graph()
        # Neural network is a Multi-Layered perceptron with one hidden layer containing tanh units
        with self.graph.as_default():
            tf.set_random_seed(1234)
            self.weights = {
                'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
                'out': tf.Variable(tf.random_normal([self.n_hidden_1, 1]))
            }
            self.biases = {
                'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                'out': tf.Variable(tf.random_normal([1]))
            }
            self.state_input = self.x = tf.placeholder("float", [None, len(self.observation_space.high)])  # State input
            self.return_input = tf.placeholder("float")  # Target return
            self.value_pred = self.multilayer_perceptron(self.state_input, self.weights, self.biases)
            self.loss = tf.reduce_mean(tf.pow(self.value_pred - self.return_input, 2))
            self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            init = tf.initialize_all_variables()
        print("Value Graph Constructed")
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(init)

    def multilayer_perceptron(self, x, weights, biases):
        """Constructs the multilayere perceptron model"""
        # First hidden layer
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.tanh(layer_1)
        # Output Layer
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer

    def update_value_estimate(self):
        """Uses mini batch gradient descent to update the value estimate"""
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
        # Monte Carlo prediction
        batch_size = self.batch_size
        if np.ma.size(replay_states) < batch_size:
            batch_size = np.ma.size(replay_states)

        for epoch in range(self.num_epochs):
            total_batch = int(np.ma.size(replay_states) / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_state_input, batch_return_input = self.get_next_batch(batch_size, replay_states,
                                                                            replay_return_from_states)
                # Fit training data using batch
                self.sess.run(self.optim,
                              feed_dict={self.state_input: batch_state_input, self.return_input: batch_return_input})

    def get_advantage_vector(self, states, rewards, next_states):
        """Returns TD(0) Advantage for particular state and action"""

        advantage_vector = []
        for i in range(len(states)):
            state = np.asarray(states[i])
            state = state.reshape(1, len(self.observation_space.high))
            next_state = np.asarray(next_states[i])
            next_state = next_state.reshape(1, len(self.observation_space.high))
            reward = rewards[i]
            state_value = self.sess.run(self.value_pred, feed_dict={self.state_input: state})
            next_state_value = self.sess.run(self.value_pred, feed_dict={self.state_input: next_state})
            # This follows directly from the forula for TD(0)
            advantage = reward + self.discount * next_state_value - state_value
            advantage_vector.append(advantage)

        return advantage_vector

    def get_next_batch(self, batch_size, states_data, returns_data):
        """Return mini-batch of transitions from replay data sampled with replacement"""
        all_states = []
        all_returns = []
        for i in range(len(states_data)):
            episode_states = states_data[i]
            episode_returns = returns_data[i]
            for j in range(len(episode_states)):
                all_states.append(episode_states[j])
                all_returns.append(episode_returns[j])
        all_states = np.asarray(all_states)
        all_returns = np.asarray(all_returns)
        randidx = np.random.randint(all_states.shape[0], size=batch_size)
        batch_states = all_states[randidx, :]
        batch_returns = all_returns[randidx]
        return batch_states, batch_returns


class ActorCriticLearner:
    def __init__(self, env, max_episodes, episodes_before_update, discount, w_rollouts=True, logger=True):
        self.env = env
        self.actor = Actor(self.env, discount, learning_rate=0.001, w_rollouts=w_rollouts)
        self.critic = Critic(self.env, discount)
        self.last_episode = 0
        self.logger = logger

        # Learner parameters
        self.max_episodes = max_episodes
        self.episodes_before_update = episodes_before_update

    def learn(self, max_env_time_steps, goal_avg_score):
        state_action_history = []
        advantage_vectors = []
        sum_reward = 0
        latest_rewards = []
        update = True
        for i in range(self.max_episodes):
            self.last_episode = i
            episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states, episode_total_reward = self.actor.rollout_policy(
                max_env_time_steps)
            advantage_vector = self.critic.get_advantage_vector(episode_states, episode_rewards, episode_next_states)
            advantage_vectors.append(advantage_vector)
            for e in range(len(episode_states)):
                # if episode_rewards[e] != 1.0:
                #     print("YES:", episode_rewards[e], episode_states[e], episode_next_states[e], episode_actions[e])
                state_action_history.append([episode_states[e], episode_actions[e], episode_next_states[e], episode_rewards[e]])
            latest_rewards.append(episode_total_reward)
            if len(latest_rewards) > 100:
                latest_rewards.pop(0)
            sum_reward += episode_total_reward
            if (i + 1) % self.episodes_before_update == 0:
                avg_reward = sum_reward / self.episodes_before_update
                if self.logger:
                    print("Current {} episode average reward: {}".format(i, avg_reward))
                # In this part of the code I try to reduce the effects of randomness leading to oscillations in my
                # network by sticking to a solution if it is close to final solution.
                # If the average reward for past batch of episodes exceeds that for solving the environment, continue with it
                if avg_reward >= goal_avg_score:  # This is the criteria for having solved the environment by Open-AI Gym
                    update = False
                else:
                    update = True

                if update:
                    if self.logger:
                        print("Updating")
                    self.actor.update_policy(advantage_vectors)
                    self.critic.update_value_estimate()
                else:
                    if self.logger:
                        print("Good Solution, not updating")
                # Delete the data collected so far
                del advantage_vectors[:]
                self.actor.reset_memory()
                sum_reward = 0

                avg_rew = sum(latest_rewards) / float(len(latest_rewards))
                if self.logger:
                    print("Episode:", i, " - AVG:", avg_rew)
                if avg_rew >= goal_avg_score and len(latest_rewards) >= 100:
                    if self.logger:
                        print("Avg reward over", goal_avg_score, ":", avg_rew)
                    break
        return state_action_history
