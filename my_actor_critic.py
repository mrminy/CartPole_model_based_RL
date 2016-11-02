"""
This is an actor-critic implementation to experiment with model-based RL.
The code is a modification from mohakbhardwaj's version on gym.openai.com.
The transition model is pre-learned by gathered data and supervised learning. See model_based_learner.py
Link: https://gym.openai.com/evaluations/eval_KhmXmmgmSManWEtxZJoLeg
Github: https://gist.github.com/mohakbhardwaj/3d895b41efbceff93874228cc0f39132#file-cartpole-policy-gradient-py
"""

import random
import numpy as np
import tensorflow as tf

import model_based_learner


class Actor:
    def __init__(self, env, transition_model, imagination_rollouts=0, action_uncertainty=0.0, discount=0.90,
                 learning_rate=0.01):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.action_space_n = self.action_space.n
        self.action_uncertainty = action_uncertainty
        # Learning parameters
        self.learning_rate = learning_rate
        self.imagination_rollouts = imagination_rollouts
        self.imagination_learning_rate = learning_rate
        self.discount = discount
        self.max_reward_for_game = -99999999.99
        # Declare tf graph
        self.graph = tf.Graph()

        self.transition_model = transition_model

        # Build the graph when instantiated
        with self.graph.as_default():
            tf.set_random_seed(1234)
            self.weights = tf.Variable(tf.random_normal([len(self.observation_space.high), self.action_space_n]), name='weights')
            self.biases = tf.Variable(tf.random_normal([self.action_space_n]))

            # Neural Network inputs
            # The types of inputs possible include: state, advantage, action(to return probability of executing that action)
            self.x = tf.placeholder("float", [None, len(self.observation_space.high)])  # State input
            self.y = tf.placeholder("float")  # Advantage input
            self.action_input = tf.placeholder("float", [None,
                                                         self.action_space_n])  # Input action to return the probability associated with that action
            loss_const = tf.constant(0.00001)

            # Current policy is a simple softmax policy since actions are discrete in this environment
            self.policy = self.softmax_policy(self.x, self.weights, self.biases)  # Softmax policy
            # The following are derived directly from the formula for gradient of policy
            self.log_action_probability = tf.reduce_sum(self.action_input * tf.log(self.policy + loss_const))
            self.loss = -self.log_action_probability * self.y  # Loss is score function times advantage
            # Use Adam Optimizer to optimize

            self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            # self.optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            # Initializing all variables
            self.init = tf.initialize_all_variables()
            print("Policy Graph Constructed")

        # Declare a TF session and initialize it
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def rollout_policy(self, timeSteps, explore):
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
            # Choose selected_action based on current policy
            selected_action, executed_action = self.choose_action(curr_state, explore)

            # Execute the selected_action in the environment and observe reward
            next_state, reward, done, info = self.env.step(executed_action)
            # Update the total reward
            total_reward += reward

            if done or time >= self.env.spec.timestep_limit:
                # Skip training when done or time-step is above the limit of the env
                break

            # Add state, selected_action, reward transitions to containers for episode data
            curr_state_l = curr_state.tolist()
            next_state_l = next_state.tolist()
            if curr_state_l not in episode_states:
                episode_states.append(curr_state_l)
                episode_actions.append(selected_action)
                episode_rewards.append(reward)
                episode_next_states.append(next_state_l)
                episode_return_from_states.append(reward)
                for i in range(len(episode_return_from_states) - 1):
                    # Here multiply the reward by discount factor raised to the power len(episode_return_from_states)-1-i
                    episode_return_from_states[i] += pow(self.discount,
                                                         len(episode_return_from_states) - 1 - i) * reward
            else:
                # Iterate through the replay memory and update the final return for all states, i.e don't add the
                # state if it is already there but update reward for other states
                for i in range(len(episode_return_from_states)):
                    episode_return_from_states[i] += pow(self.discount,
                                                         len(episode_return_from_states) - i) * reward

            curr_state = next_state
        if total_reward > self.max_reward_for_game:
            self.max_reward_for_game = total_reward

        # Update the global replay memory
        self.update_memory(episode_states, episode_actions, episode_rewards, episode_next_states,
                           episode_return_from_states, True)
        return episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states, total_reward

    def perform_imagination_rollouts(self, time_steps, start_state):
        """Rollout policy for one episode, update the replay memory and return total reward"""
        total_reward = 0
        curr_state = start_state
        prev_state = curr_state
        # Initialize lists in order to store episode data
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        episode_return_from_states = []

        for time in range(time_steps):
            # Choose selected_action based on current policy
            selected_action, executed_action = self.choose_action(curr_state, True)
            # Execute the selected_action in the environment and observe reward
            next_state, reward, done, _ = self.transition_model.predict(curr_state, executed_action)

            # Update the total reward
            total_reward += reward
            if done or time >= self.env.spec.timestep_limit:
                break

            # Add state, selected_action, reward transitions to containers for episode data
            curr_state_l = curr_state.tolist()
            next_state_l = next_state.tolist()
            if curr_state_l not in episode_states:
                episode_states.append(curr_state_l)
                episode_actions.append(selected_action)
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

        # # Update the global replay memory from rollout experience
        # self.update_memory(rollout_episode_states, rollout_episode_actions, rollout_episode_rewards,
        #                    rollout_episode_next_states, rollout_episode_return_from_states)
        self.update_memory(episode_states, episode_actions, episode_rewards, episode_next_states,
                           episode_return_from_states, False)
        return episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states, total_reward

    def update_policy(self, advantage_vectors):
        """Updates the policy weights by running gradient descent on one state at a time"""
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states

        errors = []
        for i in range(len(replay_states)):
            states = replay_states[i]
            actions = replay_actions[i]
            advantage_vector = advantage_vectors[i]
            for j in range(len(states)):
                action = self.to_action_input(actions[j])

                state = np.asarray(states[j])
                state = state.reshape(1, len(self.observation_space.high))

                o, error_value = self.sess.run([self.optim, self.loss],
                                               feed_dict={self.x: state, self.action_input: action,
                                                          self.y: advantage_vector[j]})
                errors.append(error_value)
        # print("E:", errors)
        # print("AV:", advantage_vectors)

    def softmax_policy(self, state, weights, biases):
        """Defines softmax policy for tf graph"""
        policy = tf.nn.softmax(tf.matmul(state, weights) + biases)
        return policy

    def choose_action(self, state, explore=True):
        """Chooses action from the crrent policy and weights"""
        state = np.asarray(state)
        state = state.reshape(1, len(self.observation_space.high))
        softmax_out = self.sess.run(self.policy, feed_dict={self.x: state})

        # print("2", state)
        # if random.random() < 0.01:
        #     print("3", softmax_out)
        # if random.random() < 0.01:
        #     print("4.5", self.sess.run(self.weights))
        if np.math.isnan(softmax_out[0][0]):
            self.sess.close()
            exit()
        if explore:
            # Sample action from prob density
            action = np.random.choice(np.arange(max(2, self.action_space_n)), 1, replace=True, p=softmax_out[0])[0]
            # print(action, softmax_out)
            # action = np.random.choice([0, 1], 1, replace=True, p=softmax_out[0])[0]
        else:
            # Follow optimal policy (argmax)
            action = np.argmax(softmax_out[0])

        # Action uncertainty (makes the environment non-discrete)
        exec_action = action
        if random.random() < self.action_uncertainty:
            exec_action = np.random.choice(self.env.action_space.n)
        return action, exec_action

    def update_memory(self, episode_states, episode_actions, episode_rewards, episode_next_states,
                      episode_return_from_states, real_data):
        """Updates the global replay memory"""
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states, all_experience
        # Using first visit Monte Carlo so total return from a state is calculated from first time it is visited

        replay_states.append(episode_states)
        replay_actions.append(episode_actions)
        replay_rewards.append(episode_rewards)
        replay_next_states.append(episode_next_states)
        replay_return_from_states.append(episode_return_from_states)
        if real_data:
            for i in range(len(episode_states)):
                all_experience.append(
                    [episode_states[i], episode_actions[i], episode_next_states[i], episode_rewards[i]])

    def reset_memory(self):
        """Resets the global replay memory"""
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
        del replay_states[:], replay_actions[:], replay_rewards[:], replay_next_states[:], replay_return_from_states[:]

    def to_action_input(self, action):
        """Utility function to convert action to a format suitable for the neura networ input"""
        action_input = [0] * self.action_space_n
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
        if batch_size != 0:
            for epoch in range(self.num_epochs):
                total_batch = int(np.ma.size(replay_states) / batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_state_input, batch_return_input = self.get_next_batch(batch_size, replay_states,
                                                                                replay_return_from_states)
                    # Fit training data using batch
                    self.sess.run(self.optim,
                                  feed_dict={self.state_input: batch_state_input,
                                             self.return_input: batch_return_input})
        else:
            print("ERROR: batch_size == 0", batch_size, len(replay_states))

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
    def __init__(self, env, max_episodes, episodes_before_update, discount, n_pre_training_epochs=100,
                 n_rollout_epochs=5, action_uncertainty=0.0,
                 logger=True):
        self.env = env
        self.transition_model = model_based_learner.TF_Transition_model(env)
        # self.transition_model.restore_model(restore_path=transition_model_restore_path)
        self.actor = Actor(self.env, self.transition_model, n_rollout_epochs, action_uncertainty=action_uncertainty,
                           discount=discount, learning_rate=0.01)
        self.critic = Critic(self.env, discount)
        self.last_episode = 0
        self.solved = False
        self.logger = logger
        self.n_pre_training_epochs = n_pre_training_epochs
        self.n_rollout_epochs = n_rollout_epochs

        # Learner parameters
        self.max_episodes = max_episodes
        self.episodes_before_update = episodes_before_update

        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states, all_experience
        replay_states = []
        replay_actions = []
        replay_rewards = []
        replay_next_states = []
        replay_return_from_states = []
        all_experience = []

    def pre_learn(self, max_env_time_steps, goal_avg_score, n_epochs=1, logger=True):
        state_action_history = []
        advantage_vectors = []
        sum_reward = 0
        latest_rewards = []
        old_lr = self.actor.learning_rate
        self.actor.learning_rate = self.actor.imagination_learning_rate

        for i in range(0, n_epochs):
            episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states, episode_total_reward = self.actor.perform_imagination_rollouts(
                max_env_time_steps, self.env.reset())
            advantage_vector = self.critic.get_advantage_vector(episode_states, episode_rewards, episode_next_states)
            advantage_vectors.append(advantage_vector)
            for e in range(len(episode_states)):
                state_action_history.append(
                    [episode_states[e], episode_actions[e], episode_next_states[e], episode_rewards[e]])
            latest_rewards.append(episode_total_reward)
            if len(latest_rewards) > 100:
                latest_rewards.pop(0)
            sum_reward += episode_total_reward
            if (i + 1) % self.episodes_before_update == 0:
                avg_reward = sum_reward / self.episodes_before_update
                if logger:
                    print("Current {} episode average reward: {}".format(i, avg_reward))
                # In this part of the code I try to reduce the effects of randomness leading to oscillations in my
                # network by sticking to a solution if it is close to final solution.
                # If the average reward for past batch of episodes exceeds that for solving the environment, continue with it
                if avg_reward >= goal_avg_score:  # This is the criteria for having solved the environment by Open-AI Gym
                    update = False
                else:
                    update = True

                if update:
                    if logger:
                        print("Updating")
                    self.actor.update_policy(advantage_vectors)
                    self.critic.update_value_estimate()
                else:
                    if logger:
                        print("Good Solution, not updating")
                # Delete the data collected so far
                del advantage_vectors[:]
                self.actor.reset_memory()
                sum_reward = 0

                avg_rew = sum(latest_rewards) / float(len(latest_rewards))
                if logger:
                    print("Pretraining episode:", i, " - AVG:", avg_rew)
                if avg_rew >= goal_avg_score and len(latest_rewards) >= 100:
                    if logger:
                        print("Avg reward over", goal_avg_score, ":", avg_rew)
                    break
        self.actor.learning_rate = old_lr

    def gather_random_data(self, n_steps=1000):
        global all_experience
        total_steps = 0
        input_scale = []
        while True:
            state = self.env.reset()
            input_scale = state
            for time_step in range(self.max_episodes):
                action = np.random.choice(self.env.action_space.n)
                next_state, reward, done, info = self.env.step(action)
                all_experience.append([state, action, next_state, reward])
                for i in range(len(state)):
                    if state[i] > input_scale[i]:
                        input_scale[i] = state[i]
                    if next_state[i] > input_scale[i]:
                        input_scale[i] = next_state[i]
                state = next_state
                total_steps += 1

                if done or total_steps >= n_steps:
                    break
            if total_steps >= n_steps:
                break
        return input_scale

    def learn(self, max_env_time_steps, goal_avg_score, learning_rate=0.01, imagination_learning_rate=0.0001):
        self.actor.imagination_learning_rate = imagination_learning_rate
        self.actor.learning_rate = learning_rate

        if self.n_pre_training_epochs != 0:
            # Gathering data for imagination rollouts
            input_scale = self.gather_random_data(1000)

            # Train transition model on gathered data and pre-train actor-critic from imagination rollouts
            global all_experience
            training_data = np.array(all_experience)
            if len(training_data) >= 1000:
                print("Training size:", len(training_data))
                test_data_r = np.load('cartpole_data/random_agent/testing_data.npy')
                test_data_ac = np.load('cartpole_data/actor_critic/testing_data.npy')
                self.transition_model = model_based_learner.TF_Transition_model(self.env, display_step=500)
                acc1, acc2 = self.transition_model.train(training_epochs=3000, learning_rate=0.0005,
                                                         training_data=training_data, test_data_r=test_data_r,
                                                         test_data_ac=test_data_ac, logger=False)
                self.actor.transition_model = self.transition_model

                # Doing imagination episodes if test error is less than a threshold
                self.pre_learn(max_env_time_steps, goal_avg_score, n_epochs=self.n_pre_training_epochs)

        state_action_history = []
        advantage_vectors = []
        sum_reward = 0
        latest_rewards = []
        update = True

        for i in range(self.max_episodes):
            if not self.solved:
                self.last_episode = int(i)
            episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states, episode_total_reward = self.actor.rollout_policy(
                max_env_time_steps, update)
            advantage_vector = self.critic.get_advantage_vector(episode_states, episode_rewards, episode_next_states)
            advantage_vectors.append(advantage_vector)
            for e in range(len(episode_states)):
                state_action_history.append(
                    [episode_states[e], episode_actions[e], episode_next_states[e], episode_rewards[e]])
            latest_rewards.append(episode_total_reward)
            if len(latest_rewards) > 100:
                latest_rewards.pop(0)
            sum_reward += episode_total_reward
            if (i + 1) % self.episodes_before_update == 0:
                avg_reward = sum_reward / self.episodes_before_update
                if self.logger:
                    print("Current {} episode average reward: {}".format(i, avg_reward))

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

                if not self.solved:
                    avg_rew = sum(latest_rewards) / float(len(latest_rewards))
                if self.logger:
                    print("Episode:", i, " - AVG:", avg_rew)
                if avg_rew >= goal_avg_score and len(latest_rewards) >= 100:
                    self.solved = True
                    print("solved!!!", self.last_episode)
                    if self.logger:
                        print("Avg reward over", goal_avg_score, ":", avg_rew)
                    #break

                    # Trying with full imagination rollouts for each episode
                    # self.pre_learn(max_env_time_steps, goal_avg_score, n_epochs=self.n_rollout_epochs, logger=False)
        print("All real life steps:", len(all_experience))
        return state_action_history
