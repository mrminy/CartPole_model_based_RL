import gym
import numpy as np
import tensorflow as tf


class TF_Transition_model:
    def __init__(self, env, w_init_limit=(-0.5, 0.5)):
        self.env = env
        self.graph = tf.Graph()
        self.input_scale = [4.8, 3.58683067, 0.418879020479, 3.69079514]  # Correct scaling based on env
        # self.input_scale = [2.5, 3.58683067, 0.28, 3.69079514]  # TODO make this not fixed
        # self.input_scale = [2.46059875, 3.58683067, 0.27366452, 3.69079514] # Scaling for random agent
        # self.input_scale = [2.46059875, 3.519398, 0.27262908, 3.45911401]  # Scaling for actor critic

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
        state_representation = np.array(curr_state)
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
        testing_data_random_agent = np.load('cartpole_data/random_agent/testing_data.npy')
        testing_data_actor_critic = np.load('cartpole_data/actor_critic/testing_data.npy')
        print("Preprocessing data...")
        X_train, X_train_action, Y_train = self.preprocess_data(training_data)
        X_test_r, X_test_action_r, Y_test_r = self.preprocess_data(testing_data_random_agent)
        X_test_a_c, X_test_action_a_c, Y_test_a_c = self.preprocess_data(testing_data_actor_critic)
        X_train, Y_train = self.scale_data(X_train, Y_train)
        X_test_r, Y_test_r = self.scale_data(X_test_r, Y_test_r)
        X_test_a_c, Y_test_a_c = self.scale_data(X_test_a_c, Y_test_a_c)
        print("Maximum values found:", self.x_max)

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

                acc_random_agent = self.sess.run(self.accuracy,
                                                 feed_dict={self.X: X_test_r, self.X_action: X_test_action_r,
                                                            self.Y: Y_test_r})
                print("Test error random agent:", acc_random_agent)
                acc_actor_critic = self.sess.run(self.accuracy,
                                                 feed_dict={self.X: X_test_a_c, self.X_action: X_test_action_a_c,
                                                            self.Y: Y_test_a_c})
                print("Test error actor critic:", acc_actor_critic)

        acc_random_agent = self.sess.run(self.accuracy,
                                         feed_dict={self.X: X_test_r, self.X_action: X_test_action_r, self.Y: Y_test_r})
        print("Final test error random agent:", acc_random_agent)
        acc_actor_critic = self.sess.run(self.accuracy,
                                         feed_dict={self.X: X_test_a_c, self.X_action: X_test_action_a_c,
                                                    self.Y: Y_test_a_c})
        print("Final test error actor critic:", acc_actor_critic)

        # Applying encode and decode over test set and show some examples
        encode_decode = self.sess.run(self.y_pred, feed_dict={self.X: X_test_r[:self.examples_to_show],
                                                              self.X_action: X_test_action_r[:self.examples_to_show]})
        print(Y_test_r[:self.examples_to_show])
        print(encode_decode)

        if save:
            save_path = self.saver.save(self.sess, save_path)
            print("Model saved in file: %s" % save_path)

        return acc_random_agent, acc_actor_critic

    def scale_data(self, x, y):
        return x / self.x_max, y / self.x_max

    def preprocess_data(self, value):
        x = []
        x_action = []
        y = []
        for d in value:
            for j in range(len(d[0])):
                if abs(d[0][j]) > self.x_max[j]:
                    self.x_max[j] = abs(d[0][j])
                if abs(d[2][j]) > self.x_max[j]:
                    self.x_max[j] = abs(d[2][j])

            x.append(np.array(d[0]))  # State t (and scale)
            x_action.append([d[1]])  # Action
            y.append(np.array(d[2]))  # State t+1 (and scale)
        x, x_action, y = np.array(x), np.array(x_action), np.array(y)
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

            # Evaluate model
            # correct_pred = tf.equal(self.y_pred, y_true)
            correct_pred = tf.pow(y_true - self.y_pred, 2)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # Creates a saver
            self.saver = tf.train.Saver()

            # Initializing the variables
            self.init = tf.initialize_all_variables()

            # Launch the graph
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(self.init)


if __name__ == '__main__':
    """
    TODO
    - Try dropout
    - Try LSTM for the prediction layer (feed sequential bathces)
    """
    env = gym.make('CartPole-v0')
    model = TF_Transition_model(env, w_init_limit=(-0.2, 0.2))
    # Old x_max [2.3817147442718847, 2.9103841971082174, 0.20939462615293689, 2.6612303605854293]
    # model.train(learning_rate=0.0005, training_epochs=30,
    #             train_data_path='cartpole_data/random_agent/training_data.npy', save=True,
    #             save_path="new_transition_model/transition_model.ckpt")
    # Maximum values found with both test set: [2.46059875  3.58683067  0.27366452  3.69079514]

    model.train(learning_rate=0.0005, training_epochs=30,
                train_data_path='cartpole_data/actor_critic/training_data.npy', save=True,
                save_path="new_transition_model/transition_model_actor_critic.ckpt")
    # Maximum values found with both test set: [2.46059875  3.519398    0.27262908  3.45911401]

"""
30 epochs trained on a_c
Final test error random agent: 1.42732e-05
Final test error actor critic: 1.55221e-06
[[-0.01268554 -0.05760668 -0.04464523  0.07498149]
 [-0.01433343 -0.11299857 -0.02561795  0.15847661]
 [-0.01756588 -0.16841235  0.01459703  0.2424502 ]
 [-0.02238348 -0.22386958  0.07612113  0.32742334]
 [-0.0287875  -0.27938685  0.15920799  0.41389601]]
[[-0.01353095 -0.05727049 -0.04516043  0.07504165]
 [-0.01511772 -0.11251857 -0.02665925  0.15866895]
 [-0.0173834  -0.16769446  0.01339999  0.24265979]
 [-0.02103016 -0.22266482  0.07535179  0.32737523]
 [-0.0262944  -0.27759272  0.15892783  0.41424477]]

30 epochs trained on r
Final test error random agent: 1.41247e-06
Final test error actor critic: 4.52601e-05
[[-0.01268554 -0.05652367 -0.04447631  0.0702747 ]
 [-0.01433343 -0.11087419 -0.02552102  0.1485286 ]
 [-0.01756588 -0.16524618  0.0145418   0.22723095]
 [-0.02238348 -0.21966082  0.07583312  0.3068701 ]
 [-0.0287875  -0.27413436  0.1586056   0.38791465]]
[[-0.0133454  -0.0568892  -0.0439763   0.0692116 ]
 [-0.01477456 -0.11096619 -0.02529106  0.14747195]
 [-0.01773186 -0.16490845  0.01447422  0.22606078]
 [-0.02255355 -0.21909066  0.07545467  0.30554152]
 [-0.02929129 -0.27353701  0.15783083  0.38661772]]

AC result (modified scaling, 216 pre-training epochs, lr 0.001)
[197, 165, 225, 179, 139, 473, 927, 999, 159, 233, 221, 427, 259, 161, 689, 139, 999, 353, 185, 777]
"""
