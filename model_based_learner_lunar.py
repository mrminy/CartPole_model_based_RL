"""
This is my implementation of a deep neural network to approximate the transition function of an environment (LunarLander).
The transition model is trained supervised by previously gathered data from the environment.
The trained model is then further used to improve learning of a general agent in the same environment.
Further work could involve retraining the transition model after n real episodes (like Dyna-Q), performing shorter
 imagination rollouts in real episodes for planning trajectories, or improve exploration policy by using the
 uncertanty of the transition model.
"""

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import common


class TF_Done_model:
    """
    This model predicts the terminal state of the environment
    """

    def __init__(self, env, input_scale, history_sampling_rate=1, w_init_limit=(-0.5, 0.5), display_step=1):
        self.env = env
        self.input_scale = input_scale
        self.history_sampling_rate = history_sampling_rate
        self.w_init_limit = w_init_limit
        self.display_step = display_step
        self.examples_to_show = 0

        self.graph = tf.Graph()

        # tf placeholders
        self.X = None
        self.Y = None
        self.y_pred = None
        self.sess = None
        self.keep_prob = None  # for dropout
        self.saver = None

        # Network Parameters
        self.state_space = self.env.observation_space.shape[0]
        self.n_input = self.env.observation_space.shape[0]  # Next state
        self.n_output = 1  # Done (1) or not done (0)
        self.n_hidden_1 = 16  # 1st layer num features
        self.n_hidden_2 = 16  # 2nd layer num features

        self.cost_history = []
        self.test_acc_history = []

    def predict(self, state, action_conv):
        """
        :param state: the state of the environment
        :return: the prediction of state being a terminal state
        """
        state_representation = np.array(state)
        state_representation /= self.input_scale
        # Keep prob can be reduced a bit to generate uncertainty in model
        transition_prediction = self.sess.run(self.y_pred,
                                              feed_dict={self.X: [state_representation], self.keep_prob: 1.0})

        # Returning the predicted class [not done, done]
        done_prediction = transition_prediction[0]
        if done_prediction > 0.5:
            return True
        return False

    def restore_model(self, restore_path='done_model/tf_done_model.ckpt'):
        self.build_model()
        self.saver.restore(self.sess, restore_path)
        print("Model restored from file: %s" % restore_path)

    def build_model(self, learning_rate=0.001):
        print("Building done graph...")
        with self.graph.as_default():
            self.X = tf.placeholder("float", [None, self.state_space])  # next state input
            self.Y = tf.placeholder("float", [None, self.n_output])  # prediction done/not done
            self.keep_prob = tf.placeholder(tf.float32)  # For dropout

            weights = {
                'h1': tf.Variable(
                    tf.random_normal([self.n_input, self.n_hidden_1])),
                'h2': tf.Variable(
                    tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
                'out': tf.Variable(
                    tf.random_normal([self.n_hidden_2, self.n_output])),
            }
            biases = {
                'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
                'bout': tf.Variable(tf.random_normal([self.n_output])),
            }

            layer_1 = tf.nn.tanh(tf.add(tf.matmul(self.X, weights['h1']), biases['b1']))
            layer_1_drop = tf.nn.dropout(layer_1, self.keep_prob)  # Dropout layer
            layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1_drop, weights['h2']), biases['b2']))
            layer_2_drop = tf.nn.dropout(layer_2, self.keep_prob)  # Dropout layer
            out = tf.nn.tanh(tf.add(tf.matmul(layer_2_drop, weights['out']), biases['bout']))
            out = tf.nn.dropout(out, self.keep_prob)  # Dropout layer

            # Prediction
            self.y_pred = out
            # Targets (Labels) are the input data.
            y_true = self.Y

            # Define loss, minimize the squared error (with or without scaling)
            self.loss_function = tf.reduce_mean(tf.square(y_true - self.y_pred))
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_function)
            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.01).minimize(self.loss_function)

            # Evaluate model
            self.accuracy = tf.reduce_mean(tf.cast(self.loss_function, tf.float32))

            # Creates a saver
            self.saver = tf.train.Saver()

            # Initializing the variables
            self.init = tf.initialize_all_variables()

            # Launch the graph
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.graph, config=config)
            self.sess.run(self.init)

    def train(self, training_epochs=20, learning_rate=0.001, batch_size=128, show_cost=False, show_test_acc=False,
              save=False, training_data=None, test_data_r=None, test_data_ac=None,
              save_path='done_model/tf_done_model.ckpt', logger=True, max_training_data=999999):
        # Load and preprocess data
        if logger:
            print("Preprocessing data...")
        X_train, Y_train = self.preprocess_data(training_data, max_data=max_training_data)
        X_test_r, Y_test_r = self.preprocess_data(test_data_r)
        X_test_a_c, Y_test_a_c = self.preprocess_data(test_data_ac)
        X_train = self.scale_data(X_train, Y_train)
        X_test_r = self.scale_data(X_test_r, Y_test_r)
        X_test_a_c = self.scale_data(X_test_a_c, Y_test_a_c)

        self.build_model(learning_rate=learning_rate)

        total_batch = int(len(X_train) / batch_size)
        if logger:
            print("Starting training...")
            print("Total nr of batches:", total_batch)
        # Training the model
        for epoch in range(training_epochs):
            # Loop over all batches
            c = None
            idexes = np.arange(len(X_train))
            for i in range(total_batch):
                idx = np.random.choice(idexes, batch_size, replace=True)
                batch_xs = X_train[idx]
                batch_ys = Y_train[idx]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = self.sess.run([self.optimizer, self.loss_function],
                                     feed_dict={self.X: batch_xs, self.Y: batch_ys, self.keep_prob: 1.0})
                if i % self.history_sampling_rate == 0:
                    self.cost_history.append(c)
                    if len(X_test_r) > 0:
                        sampled_indexes = np.random.choice(np.arange(0, len(X_test_r)), 50000)
                        self.test_acc_history.append(self.sess.run(self.accuracy,
                                                                   feed_dict={self.X: X_test_r[sampled_indexes],
                                                                              self.Y: Y_test_r[sampled_indexes],
                                                                              self.keep_prob: 1.0}))

            # Display logs per epoch step
            test_error = 0.0
            if epoch % self.display_step == 0 and c is not None and len(X_test_r) > 0:
                test_error = self.sess.run(self.accuracy, feed_dict={self.X: X_test_r[sampled_indexes],
                                                                     self.Y: Y_test_r[sampled_indexes],
                                                                     self.keep_prob: 1.0})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "test error=",
                  "{:.9f}".format(test_error))

        acc_random_agent = 0.0
        acc_actor_critic = 0.0
        if len(X_test_r) > 0:
            acc_random_agent = self.sess.run(self.accuracy,
                                             feed_dict={self.X: X_test_r, self.Y: Y_test_r,
                                                        self.keep_prob: 1.0})
            print("Final test error random agent:", acc_random_agent)

        if len(X_test_a_c) > 0 and len(Y_test_a_c) > 0:
            acc_actor_critic = self.sess.run(self.accuracy,
                                             feed_dict={self.X: X_test_a_c, self.Y: Y_test_a_c, self.keep_prob: 1.0})
            print("Final test error actor critic:", acc_actor_critic)

        if len(X_test_r) > 0:
            # Applying prediction over test set and show some examples
            prediction = self.sess.run(self.y_pred,
                                       feed_dict={self.X: X_test_r[:self.examples_to_show], self.keep_prob: 1.0})
            print(Y_test_r[:self.examples_to_show])
            print(prediction[:])

        if save:
            save_path = self.saver.save(self.sess, save_path)
            print("Model saved in file: %s" % save_path)

        if show_test_acc:
            y_axis = np.array(self.test_acc_history)
            plt.plot(y_axis)
            plt.show()

        if show_cost:
            y_axis = np.array(self.cost_history)
            plt.plot(y_axis)
            plt.show()

        return acc_random_agent, acc_actor_critic

    def scale_data(self, x, y):
        if len(x) > 0:
            return x / self.input_scale
        return x

    def preprocess_data(self, value, max_data=999999999):
        x = []
        y = []
        if value is None or len(value) == 0:
            return np.array(x), np.array(y)
        for d in value:
            x.append(np.array(d[2]))  # Next state (t+1)
            if d[4]:
                y.append(np.array([1.]))  # Done
            else:
                y.append(np.array([0.]))
            if len(y) >= max_data:
                break
        x, y = np.array(x), np.array(y)
        return x, y


class TF_Reward_model:
    """
    This model predicts the reward received after performing an action 'a' in a state 'St' and observing the next
    predicted state 'St+1'
    """

    def __init__(self, env, input_scale, history_sampling_rate=1, w_init_limit=(-0.5, 0.5), display_step=1):
        self.env = env
        self.max_reward = env.spec.reward_threshold  # This is updated during the pre-processing
        if self.max_reward is None:
            self.max_reward = -999999999.99
        self.input_scale = input_scale
        self.history_sampling_rate = history_sampling_rate
        self.w_init_limit = w_init_limit
        self.display_step = display_step
        self.examples_to_show = 200
        self.training_epoch = 0

        self.graph = tf.Graph()

        # tf placeholders
        self.X = None
        self.X_action = None
        self.Y = None
        self.y_pred = None
        self.sess = None
        self.keep_prob = None  # for dropout
        self.saver = None

        # Network Parameters
        self.n_action = common.get_n_action(self.env.action_space.n)
        self.state_space = self.env.observation_space.shape[0]
        self.n_input = self.env.observation_space.shape[0]  # Next state
        self.n_output = 1  # Predicted reward
        self.n_hidden_1 = 16  # 1st layer num features
        self.n_hidden_2 = 16  # 2nd layer num features

        self.cost_history = []
        self.test_acc_history = []

    def predict(self, curr_state, action_conv):
        """
        :param curr_state: the current state of the environment
        :param action: the selected action to take for this state
        :return: the predicted next state
        """
        state_representation = np.array(curr_state)
        state_representation /= self.input_scale
        # Keep prob can be reduced a bit to generate uncertainty in model
        transition_prediction = self.sess.run(self.y_pred,
                                              feed_dict={self.X: [state_representation], self.X_action: [action_conv],
                                                         self.keep_prob: 1.0})

        # Returning the predicted transition (next state) rescaled back to normal
        reward = transition_prediction[0] * self.max_reward
        return reward[0]

    def restore_model(self, restore_path='reward_model/tf_reward_model.ckpt'):
        self.build_model()
        self.saver.restore(self.sess, restore_path)
        print("Model restored from file: %s" % restore_path)

    def build_model(self, learning_rate=0.001):
        print("Building reward graph...")
        with self.graph.as_default():
            # Encode curr_state, add transition prediction with selected action and decode to predicted output state
            self.X = tf.placeholder("float", [None, self.state_space])  # current state input
            self.X_action = tf.placeholder("float", [None, self.n_action])  # action input
            self.Y = tf.placeholder("float", [None, self.n_output])  # output
            self.keep_prob = tf.placeholder(tf.float32)  # For dropout

            weights = {
                'h1': tf.Variable(
                    tf.random_normal([self.n_input, self.n_hidden_1])),
                'h2': tf.Variable(
                    tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
                'out': tf.Variable(
                    tf.random_normal([self.n_hidden_2, self.n_output])),
            }
            biases = {
                'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
                'bout': tf.Variable(tf.random_normal([self.n_output])),
            }

            layer_1 = tf.nn.tanh(tf.add(tf.matmul(self.X, weights['h1']), biases['b1']))
            layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
            out = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['out']), biases['bout']))
            out = tf.nn.dropout(out, self.keep_prob)  # Dropout layer

            # Prediction
            self.y_pred = out
            # Targets (Labels) are the input data.
            y_true = self.Y

            # Define loss, minimize the squared error (with or without scaling)
            # self.loss_function = tf.reduce_mean(tf.pow(((y_true - self.y_pred) * self.input_scale), 2))
            self.loss_function = tf.reduce_mean(tf.square(y_true - self.y_pred))
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_function)
            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.01).minimize(self.loss_function)

            # Evaluate model
            self.accuracy = tf.reduce_mean(tf.cast(self.loss_function, tf.float32))

            # Creates a saver
            self.saver = tf.train.Saver()

            # Initializing the variables
            self.init = tf.initialize_all_variables()

            # Launch the graph
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.graph, config=config)
            self.sess.run(self.init)

    def train(self, training_epochs=20, learning_rate=0.001, batch_size=128, show_cost=False, show_test_acc=False,
              save=False, training_data=None, test_data_r=None, test_data_ac=None,
              save_path='transition_model/tf_transition_model.ckpt', logger=True, max_training_data=99999999):
        self.training_epoch = 0

        # Load and preprocess data
        if logger:
            print("Preprocessing data...")
        X_train, X_train_action, Y_train = self.preprocess_data(training_data, max_data=max_training_data)
        X_test_r, X_test_action_r, Y_test_r = self.preprocess_data(test_data_r)
        X_test_a_c, X_test_action_a_c, Y_test_a_c = self.preprocess_data(test_data_ac)
        X_train, Y_train = self.scale_data(X_train, Y_train)
        X_test_r, Y_test_r = self.scale_data(X_test_r, Y_test_r)
        X_test_a_c, Y_test_a_c = self.scale_data(X_test_a_c, Y_test_a_c)

        self.build_model(learning_rate=learning_rate)

        total_batch = int(len(X_train) / batch_size)
        if logger:
            print("Starting training...")
            print("Total nr of batches:", total_batch)
        # Training the reward model
        for epoch in range(training_epochs):
            self.training_epoch = epoch
            # Loop over all batches
            c = None
            idexes = np.arange(len(X_train))
            for i in range(total_batch):
                idx = np.random.choice(idexes, batch_size, replace=True)
                batch_xs = X_train[idx]
                batch_x_action = X_train_action[idx]
                batch_ys = Y_train[idx]
                _, c = self.sess.run([self.optimizer, self.loss_function],
                                     feed_dict={self.X: batch_xs, self.X_action: batch_x_action, self.Y: batch_ys,
                                                self.keep_prob: 1.})
                if i % self.history_sampling_rate == 0:
                    self.cost_history.append(c)
                    if len(X_test_r) > 0:
                        sampled_indexes = np.random.choice(np.arange(0, len(X_test_r)), 50000)
                        self.test_acc_history.append(self.sess.run(self.accuracy,
                                                                   feed_dict={self.X: X_test_r[sampled_indexes],
                                                                              self.X_action: X_test_action_r[
                                                                                  sampled_indexes],
                                                                              self.Y: Y_test_r[sampled_indexes],
                                                                              self.keep_prob: 1.0}))

            # Display logs per epoch step
            test_error = 0.0
            if epoch % self.display_step == 0 and c is not None and len(X_test_r) > 0:
                test_error = self.sess.run(self.accuracy, feed_dict={self.X: X_test_r[sampled_indexes],
                                                                     self.X_action: X_test_action_r[sampled_indexes],
                                                                     self.Y: Y_test_r[sampled_indexes],
                                                                     self.keep_prob: 1.0})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "test error=",
                  "{:.9f}".format(test_error))
        acc_random_agent = None
        if len(X_test_r) > 0:
            acc_random_agent = self.sess.run(self.accuracy,
                                             feed_dict={self.X: X_test_r, self.X_action: X_test_action_r,
                                                        self.Y: Y_test_r,
                                                        self.keep_prob: 1.0})
            print("Final test error random agent:", acc_random_agent)

        acc_actor_critic = 0.0
        if len(X_test_a_c) > 0 or len(Y_test_a_c) != 0:
            acc_actor_critic = self.sess.run(self.accuracy,
                                             feed_dict={self.X: X_test_a_c, self.X_action: X_test_action_a_c,
                                                        self.Y: Y_test_a_c, self.keep_prob: 1.0})
            print("Final test error actor critic:", acc_actor_critic)

        # Applying encode and decode over test set and show some examples
        if len(X_test_r) > 0:
            encode_decode = self.sess.run(self.y_pred, feed_dict={self.X: X_test_r[:self.examples_to_show],
                                                                  self.X_action: X_test_action_r[
                                                                                 :self.examples_to_show],
                                                                  self.keep_prob: 1.0})
            print(np.array(Y_test_r[:self.examples_to_show] * self.max_reward).flatten())
            print(np.array(encode_decode[:] * self.max_reward).flatten())

        if save:
            save_path = self.saver.save(self.sess, save_path)
            print("Model saved in file: %s" % save_path)

        if show_test_acc:
            y_axis = np.array(self.test_acc_history)
            plt.plot(y_axis)
            plt.show()

        if show_cost:
            y_axis = np.array(self.cost_history)
            plt.plot(y_axis)
            plt.show()

        return acc_random_agent, acc_actor_critic

    def scale_data(self, x, y):
        if len(x) == 0 or len(y) == 0:
            return x, y
        return x / self.input_scale, y / self.max_reward

    def preprocess_data(self, value, max_data=999999999):
        x = []
        x_action = []
        y = []
        if value is None or len(value) == 0:
            return np.array(x), np.array(x_action), np.array(y)
        for d in value:
            if abs(d[3]) > self.max_reward:
                self.max_reward = abs(d[3])
            x.append(np.array(d[2]))  # Next state (t+1)
            if self.n_action > 1:
                ac = np.zeros(self.n_action)
                ac[int(d[1])] = 1.0
                x_action.append(ac)  # Action
            else:
                x_action.append([d[1]])
            y.append(np.array([d[3]]))  # Reward
            if len(y) >= max_data:
                break
        x, x_action, y = np.array(x), np.array(x_action), np.array(y)
        return x, x_action, y


class TF_Transition_model:
    """
    This is the main transition model, containing the reward prediction model and terminal state prediction model as
    well as the state transition prediction model.
    """

    def __init__(self, env, history_sampling_rate=1, w_init_limit=(-0.5, 0.5), display_step=1):
        self.env = env
        self.graph = tf.Graph()
        # The only hardcoded feature for this transition model is the input scaling
        # This could easily be made into a non-hardcoded feature
        self.input_scale = [1.5, 6., 6., 4.0, 12., 9., 1., 1.]  # Scaling for LunarLander
        self.reward_model = TF_Reward_model(env, self.input_scale, history_sampling_rate=history_sampling_rate,
                                            w_init_limit=w_init_limit, display_step=display_step)
        self.done_model = TF_Done_model(env, self.input_scale, history_sampling_rate=history_sampling_rate,
                                        w_init_limit=w_init_limit, display_step=display_step)

        # Training parameters
        self.display_step = display_step
        self.examples_to_show = 5
        self.w_init_limit = w_init_limit  # The limit of the initialization of the weights

        # tf placeholders
        self.X = None
        self.X_action = None
        self.Y = None
        self.sess = None
        self.keep_prob = None  # for dropout

        # Network Parameters
        self.n_input = self.env.observation_space.shape[0]  # Prev state
        if self.env.action_space.n <= 2:
            self.n_action = 1
        else:
            self.n_action = self.env.action_space.n
        self.n_output = self.n_input  # Predicted next state
        self.n_hidden_1 = 40  # 1st layer num features
        self.n_hidden_2 = 40  # 2nd layer num features
        self.n_hidden_transition = 40  # Transition prediction layer

        self.cost_history = []
        self.test_acc_history = []
        self.history_sampling_rate = history_sampling_rate

    def predict(self, curr_state, action):
        """
        :param curr_state: the current state of the environment
        :param action: the selected action to take for this state
        :return: the predicted next state
        """
        assert len(curr_state) == self.n_input
        state_representation = np.array(curr_state)
        state_representation /= self.input_scale

        if self.env.action_space.n <= 2:
            action_conv = [action]  # Binary action input
        else:
            action_conv = np.zeros(self.env.action_space.n)
            action_conv[action] = 1.0  # Hot shot action input

        # Keep prob can be reduced a bit to generate uncertainty in model
        transition_prediction = self.sess.run(self.y_pred,
                                              feed_dict={self.X: [state_representation], self.X_action: [action_conv],
                                                         self.keep_prob: 1.0})

        # Returning the predicted transition (next state) rescaled back to normal
        next_state = transition_prediction[0] * self.input_scale
        reward = self.reward_model.predict(next_state, action_conv)

        done = self.done_model.predict(next_state, action_conv)
        return next_state, reward, done, None

    def restore_model(self, restore_path='transition_model/tf_transition_model.ckpt'):
        self.build_model()
        self.saver.restore(self.sess, restore_path)
        print("Model restored from file: %s" % restore_path)

    def build_model(self, learning_rate=0.001):
        print("Building transition graph...")
        with self.graph.as_default():
            # Encode curr_state, add transition prediction with selected action and decode to predicted output state
            self.X = tf.placeholder("float", [None, self.n_input])  # current state input
            self.X_action = tf.placeholder("float", [None, self.n_action])  # action input
            self.Y = tf.placeholder("float", [None, self.n_output])  # output
            self.keep_prob = tf.placeholder(tf.float32)  # For dropout

            weights = {
                'encoder_h1': tf.Variable(
                    tf.random_uniform([self.n_input, self.n_hidden_1], minval=self.w_init_limit[0],
                                      maxval=self.w_init_limit[1])),
                'encoder_h2': tf.Variable(
                    tf.random_uniform([self.n_hidden_1, self.n_hidden_2], minval=self.w_init_limit[0],
                                      maxval=self.w_init_limit[1])),
                'transition_h1': tf.Variable(
                    tf.random_uniform([self.n_hidden_1 + self.n_action, self.n_hidden_transition],
                                      minval=self.w_init_limit[0],
                                      maxval=self.w_init_limit[1])),
                'decoder_h1': tf.Variable(
                    tf.random_uniform([self.n_hidden_transition, self.n_output], minval=self.w_init_limit[0],
                                      maxval=self.w_init_limit[1])),
                'decoder_h2': tf.Variable(
                    tf.random_uniform([self.n_hidden_1, self.n_output], minval=self.w_init_limit[0],
                                      maxval=self.w_init_limit[1])),
            }
            biases = {
                'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
                'transition_b1': tf.Variable(tf.random_normal([self.n_hidden_transition])),
                'decoder_b1': tf.Variable(tf.random_normal([self.n_output])),
                'decoder_b2': tf.Variable(tf.random_normal([self.n_output])),
            }

            # Building the encoder
            def encoder(x):
                layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']),
                                            biases['encoder_b1']))
                layer_1_drop = tf.nn.dropout(layer_1, self.keep_prob)  # Dropout layer
                return layer_1_drop

            # Building the transition network between the encoder and decoder
            def transition(x, x_action):
                x = tf.concat(1, [x, x_action])
                layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['transition_h1']),
                                            biases['transition_b1']))
                layer_1_drop = tf.nn.dropout(layer_1, self.keep_prob)  # Dropout layer
                return layer_1_drop

            # Building the decoder
            def decoder(x):
                layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']),
                                            biases['decoder_b1']))
                layer_1_drop = tf.nn.dropout(layer_1, self.keep_prob)  # Dropout layer
                return layer_1_drop

            # Construct model
            encoder_op = encoder(self.X)
            transition_pred = transition(encoder_op, self.X_action)
            decoder_op = decoder(transition_pred)

            # Prediction
            self.y_pred = decoder_op
            # Targets (Labels) are the input data.
            y_true = self.Y

            # Define loss, minimize the squared error (with or without scaling)
            # self.loss_function = tf.reduce_mean(tf.pow(((y_true - self.y_pred) * self.input_scale), 2))
            self.loss_function = tf.reduce_mean(tf.pow(y_true - self.y_pred, 2))
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_function)

            # Evaluate model
            correct_pred = tf.equal(self.y_pred, y_true)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # Creates a saver
            self.saver = tf.train.Saver()

            # Initializing the variables
            self.init = tf.initialize_all_variables()

            # Launch the graph
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.graph, config=config)
            self.sess.run(self.init)

    def train(self, training_epochs=20, learning_rate=0.001, batch_size=128, show_cost=False, show_test_acc=False,
              save=False, training_data=None, test_data_r=None, test_data_ac=None,
              save_path='transition_model/tf_transition_model.ckpt', logger=True, max_training_data=9999999999,
              train_full_model=True):

        if train_full_model:
            # Train reward model
            self.reward_model.train(training_epochs, learning_rate, batch_size, training_data=training_data,
                                    test_data_r=test_data_r, test_data_ac=test_data_ac,
                                    logger=logger, max_training_data=max_training_data)
            # Train done model
            self.done_model.train(training_epochs, learning_rate, batch_size, training_data=training_data,
                                  test_data_r=test_data_r, test_data_ac=test_data_ac,
                                  logger=logger, max_training_data=max_training_data)

        if logger:
            print("Preprocessing data...")
        X_train, X_train_action, Y_train = self.preprocess_data(training_data, max_data=max_training_data)
        X_test_r, X_test_action_r, Y_test_r = self.preprocess_data(test_data_r)
        X_test_a_c, X_test_action_a_c, Y_test_a_c = self.preprocess_data(test_data_ac)
        X_train, Y_train = self.scale_data(X_train, Y_train)
        X_test_r, Y_test_r = self.scale_data(X_test_r, Y_test_r)
        X_test_a_c, Y_test_a_c = self.scale_data(X_test_a_c, Y_test_a_c)

        self.build_model(learning_rate=learning_rate)

        total_batch = int(len(X_train) / batch_size)
        if logger:
            print("Starting training...")
            print("Total nr of batches:", total_batch)
        # Training cycle
        for epoch in range(training_epochs):
            # Loop over all batches
            c = None
            idexes = np.arange(len(X_train))
            for i in range(total_batch):
                idx = np.random.choice(idexes, batch_size, replace=True)
                batch_xs = X_train[idx]
                batch_x_action = X_train_action[idx]
                batch_ys = Y_train[idx]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = self.sess.run([self.optimizer, self.loss_function],
                                     feed_dict={self.X: batch_xs, self.X_action: batch_x_action, self.Y: batch_ys,
                                                self.keep_prob: 1.0})
                if i % self.history_sampling_rate == 0:
                    self.cost_history.append(c)
                    if len(X_test_r) > 0:
                        sampled_indexes = np.random.choice(np.arange(0, len(X_test_r)), 50000)
                        self.test_acc_history.append(self.sess.run(self.accuracy,
                                                                   feed_dict={self.X: X_test_r[sampled_indexes],
                                                                              self.X_action: X_test_action_r[
                                                                                  sampled_indexes],
                                                                              self.Y: Y_test_r[sampled_indexes],
                                                                              self.keep_prob: 1.0}))

            # Display logs per epoch step
            test_error = 0.0
            if epoch % self.display_step == 0 and c is not None and len(X_test_r) > 0:
                test_error = self.sess.run(self.accuracy, feed_dict={self.X: X_test_r[sampled_indexes],
                                                                     self.X_action: X_test_action_r[sampled_indexes],
                                                                     self.Y: Y_test_r[sampled_indexes],
                                                                     self.keep_prob: 1.0})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "test error=",
                      "{:.9f}".format(test_error))
        acc_random_agent = 0.0
        acc_actor_critic = 0.0
        if len(X_test_r) > 0:
            acc_random_agent = self.sess.run(self.accuracy,
                                             feed_dict={self.X: X_test_r, self.X_action: X_test_action_r, self.Y: Y_test_r,
                                                        self.keep_prob: 1.0})
            print("Final test error random agent:", acc_random_agent)

        if len(X_test_a_c) > 0 and len(X_test_action_a_c) > 0 and len(Y_test_a_c) > 0:
            acc_actor_critic = self.sess.run(self.accuracy,
                                             feed_dict={self.X: X_test_a_c, self.X_action: X_test_action_a_c,
                                                        self.Y: Y_test_a_c, self.keep_prob: 1.0})
            print("Final test error actor critic:", acc_actor_critic)

        if len(X_test_r) > 0:
            # Applying encode and decode over test set and show some examples
            encode_decode = self.sess.run(self.y_pred, feed_dict={self.X: X_test_r[:self.examples_to_show],
                                                                  self.X_action: X_test_action_r[:self.examples_to_show],
                                                                  self.keep_prob: 1.0})
            print(Y_test_r[:self.examples_to_show] * self.input_scale)
            print(encode_decode[:] * self.input_scale)

        if save:
            save_path = self.saver.save(self.sess, save_path)
            print("Model saved in file: %s" % save_path)

        if show_test_acc:
            y_axis = np.array(self.test_acc_history)
            plt.plot(y_axis)
            plt.show()

        if show_cost:
            y_axis = np.array(self.cost_history)
            plt.plot(y_axis)
            plt.show()

        return acc_random_agent, acc_actor_critic

    def scale_data(self, x, y):
        if len(x) > 0 or len(y) > 0:
            return x / self.input_scale, y / self.input_scale
        return x, y

    def preprocess_data(self, value, max_data=999999999):
        x = []
        x_action = []
        y = []
        if value is None or len(value) == 0:
            return np.array(x), np.array(x_action), np.array(y)
        for d in value:
            x.append(np.array(d[0]))  # State
            if self.env.action_space.n <= 2:
                x_action.append([d[1]])  # Action
            else:
                action_conv = np.zeros(self.env.action_space.n)
                action_conv[d[1]] = 1.0
                x_action.append(action_conv)
            y.append(np.array(d[2]))  # State t+1
            if len(y) >= max_data:
                break
        x, x_action, y = np.array(x), np.array(x_action), np.array(y)
        return x, x_action, y


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    training_data = np.load('lunarlander_data_done/random_agent/training_data.npy')
    test_data_r = np.load('lunarlander_data_done/random_agent/testing_data.npy')

    # Train and test the dynamics model on previously saved data
    model = TF_Transition_model(env, history_sampling_rate=1, w_init_limit=(-0.2, 0.2))
    model.train(training_epochs=50, learning_rate=0.0005, training_data=training_data, test_data_r=test_data_r,
                save=True, save_path="./transition_model_saves/lunar_lander/transition_model.ckpt",
                train_full_model=True, max_training_data=20000)


    # Train the reward model separately
    # state_scale = [1.02053351, 4.83857279, 5.08818741, 3.68802795, 10.16092682, 7.92128525, 1., 1.]
    # state_scale = [1.5, 6., 6., 4., 12., 9., 1., 1.]
    # model = TF_Reward_model(env, state_scale, history_sampling_rate=1, w_init_limit=(-0.2, 0.2))
    # model.train(training_data=training_data, test_data_r=test_data_r, test_data_ac=None, batch_size=512,
    #             training_epochs=500, learning_rate=0.0005, logger=True, save=False, show_cost=True, show_test_acc=True,
    #             save_path="reward_model.ckpt", max_training_data=10000)

    # Train the done model separately
    # model = TF_Done_model(env, state_scale, history_sampling_rate=1, w_init_limit=(-0.2, 0.2))
    # model.train(training_data=training_data, test_data_r=test_data_r, test_data_ac=None,
    #             training_epochs=500, learning_rate=0.0005, batch_size=256, logger=True, save=False, show_cost=True,
    #             show_test_acc=True, save_path="lunarlander_done_model.ckpt", max_training_data=10000)
