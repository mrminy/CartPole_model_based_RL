import gym
import numpy as np
import tensorflow as tf


class TF_Transition_model:
    def __init__(self, env, w_init_limit=(-0.5, 0.5)):
        self.env = env
        self.graph = tf.Graph()
        self.input_scale = []

        # Training parameters
        self.display_step = 1
        self.examples_to_show = 5
        self.w_init_limit = w_init_limit  # The limit of the initialization of the weights
        self.x_max = self.env.observation_space.low  # For correct scaling # TODO try with just self.env.observation_space.high

        # tf placeholders
        self.X = None
        self.X_action = None
        self.Y = None
        self.sess = None

        # Network Parameters
        self.n_input = self.env.observation_space.shape[0]  # Prev state
        self.n_action = 1  # Action # TODO add support for changing action space self.env.action_space.n (for now --> binary)
        self.n_output = self.n_input  # Predicted next state
        self.n_hidden_1 = 64  # 1st layer num features
        self.n_hidden_2 = 64  # 2nd layer num features
        self.n_hidden_transition = 128  # Transition prediction layer

    def predict(self, curr_state, action):
        """
        :param curr_state: the current state of the environment
        :param action: the selected action to take for this state
        :return: the predicted next state
        """
        assert len(curr_state) == self.n_input
        state_representation = np.array([curr_state])
        transition_prediction = self.sess.run(self.y_pred,
                                              feed_dict={self.X: [state_representation], self.X_action: [[action]]})
        # Returning the predicted transition rescaled back to normal
        return transition_prediction[0] * self.input_scale

    def restore_model(self, restore_path='transition_model/tf_transition_model.ckpt'):
        self.build_model()
        self.saver.restore(self.sess, restore_path)
        print("Model restored from file: %s" % restore_path)

    def train(self, training_epochs=20, learning_rate=0.001, batch_size=128, save=False,
              train_data_path='cartpole_data/train_reward.npy',
              test_data_path='cartpole_data/test_reward.npy',
              save_path='transition_model/tf_transition_model.ckpt', test=True):
        # Load data
        print("Loading data...")
        training_data = np.load(train_data_path)
        testing_data = np.load(test_data_path)
        print("Preprocessing data...")
        X_train, X_train_action, Y_train = self.preprocess_data(training_data)
        X_test, X_test_action, Y_test = self.preprocess_data(testing_data)

        self.build_model(learning_rate=learning_rate)

        total_batch = int(len(X_train) / batch_size)
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
                                     feed_dict={self.X: batch_xs, self.X_action: batch_x_action, self.Y: batch_ys})
            # Display logs per epoch step
            if epoch % self.display_step == 0 and c is not None:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

        if test:
            # Applying encode and decode over test set
            encode_decode = self.sess.run(self.y_pred, feed_dict={self.X: X_test[:self.examples_to_show],
                                                                  self.X_action: X_test_action[:self.examples_to_show]})
            print(Y_test[:self.examples_to_show])
            print(encode_decode)

        if save:
            save_path = self.saver.save(self.sess, save_path)
            print("Model saved in file: %s" % save_path)

    def preprocess_data(self, value):
        x = []
        x_action = []
        y = []
        for d in value:
            # d = d[0] # todo

            for j in range(len(d[0])):
                if d[0][j] > self.x_max[j]:
                    self.x_max[j] = d[0][j]
                if d[2][j] > self.x_max[j]:
                    self.x_max[j] = d[2][j]

            x.append(np.array(d[0]))  # State t (and scale)
            x_action.append([d[1]])  # Action
            y.append(np.array(d[2]))  # State t+1 (and scale)
        x, x_action, y = np.array(x), np.array(x_action), np.array(y)
        x /= self.x_max
        y /= self.x_max
        return x, x_action, y

    def build_model(self, learning_rate=0.001):
        print("Building graph...")
        with self.graph.as_default():
            # Encode curr_state, add transition prediction with selected action and decode to predicted output state
            self.X = tf.placeholder("float", [None, self.n_input])
            self.X_action = tf.placeholder("float", [None, self.n_action])
            self.Y = tf.placeholder("float", [None, self.n_output])

            weights = {
                'encoder_h1': tf.Variable(
                    tf.random_uniform([self.n_input, self.n_hidden_1], minval=self.w_init_limit[0],
                                      maxval=self.w_init_limit[1])),
                'encoder_h2': tf.Variable(
                    tf.random_uniform([self.n_hidden_1, self.n_hidden_2], minval=self.w_init_limit[0],
                                      maxval=self.w_init_limit[1])),
                'transition_h1': tf.Variable(
                    tf.random_uniform([self.n_hidden_2 + self.n_action, self.n_hidden_transition],
                                      minval=self.w_init_limit[0],
                                      maxval=self.w_init_limit[1])),
                'decoder_h1': tf.Variable(
                    tf.random_uniform([self.n_hidden_transition, self.n_hidden_1], minval=self.w_init_limit[0],
                                      maxval=self.w_init_limit[1])),
                'decoder_h2': tf.Variable(
                    tf.random_uniform([self.n_hidden_1, self.n_output], minval=self.w_init_limit[0],
                                      maxval=self.w_init_limit[1])),
            }
            biases = {
                'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
                'transition_b1': tf.Variable(tf.random_normal([self.n_hidden_transition])),
                'decoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                'decoder_b2': tf.Variable(tf.random_normal([self.n_output])),
            }

            # Building the encoder
            def encoder(x):
                layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']),
                                            biases['encoder_b1']))
                layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                            biases['encoder_b2']))
                return layer_2

            # Building the transition network between the encoder and decoder
            def transition(x, x_action):
                x = tf.concat(1, [x, x_action])
                layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['transition_h1']),
                                            biases['transition_b1']))
                return layer_1

            # Building the decoder
            def decoder(x):
                layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']),
                                            biases['decoder_b1']))
                layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                            biases['decoder_b2']))
                return layer_2

            # Construct model
            encoder_op = encoder(self.X)
            transition_pred = transition(encoder_op, self.X_action)
            decoder_op = decoder(transition_pred)

            # Prediction
            self.y_pred = decoder_op
            # Targets (Labels) are the input data.
            y_true = self.Y

            # Define loss, minimize the squared error
            self.loss_function = tf.reduce_mean(tf.pow(y_true - self.y_pred, 2))

            # L2 regularization
            # regularization = (tf.nn.l2_loss(weights['encoder_h1']) + tf.nn.l2_loss(biases['encoder_b1']) +
            #                   tf.nn.l2_loss(weights['encoder_h2']) + tf.nn.l2_loss(biases['encoder_b2']) +
            #                   tf.nn.l2_loss(weights['transition_h1']) + tf.nn.l2_loss(biases['transition_b1']) +
            #                   tf.nn.l2_loss(weights['decoder_h1']) + tf.nn.l2_loss(biases['decoder_b1']) +
            #                   tf.nn.l2_loss(weights['decoder_h2']) + tf.nn.l2_loss(biases['decoder_b2']))
            # Add the l2 loss to the loss function
            # self.loss_function += 5e-4 * regularization
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_function)

            # Creates a saver
            self.saver = tf.train.Saver()

            # Initializing the variables
            self.init = tf.initialize_all_variables()

            # Launch the graph
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(self.init)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    model = TF_Transition_model(env, w_init_limit=(-0.2, 0.2))
    # Old x_max [2.3817147442718847, 2.9103841971082174, 0.20939462615293689, 2.6612303605854293]
    model.train(learning_rate=0.0005, training_epochs=40, train_data_path='cartpole_data/random_agent/training_data.npy',
                test_data_path='cartpole_data/random_agent/testing_data.npy') # [ 2.45527371  3.35534387  0.27317711  3.60897528]
    # model.train(learning_rate=0.0005, training_epochs=20, train_data_path='cartpole_data/actor_critic/training_data.npy',
    #             test_data_path='cartpole_data/actor_critic/testing_data.npy') # [ 2.3999155   3.0620723   0.20943913  2.99131429]
