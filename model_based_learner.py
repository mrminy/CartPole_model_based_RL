import gym
import numpy as np
import tflearn
import tensorflow as tf


# from tf_transition_model import TF_Transition_model as tr_model
def get_min_max():
    """
    :return: max, min from the training dataset
    """
    # x_max = [2.3817147442718847, 2.9103841971082174, 0.20939462615293689, 2.6612303605854293]
    x_max = [2.5, 3.0, 0.25, 2.75]
    x_min = [-2.391525269742431, -2.8946690139833624, -0.20933102768952855, -2.9099185565733316]
    return x_max, x_min


def preprocess_data_transition_model(value):
    x = []
    x_action = []
    y = []
    x_max, x_min = get_min_max()
    for d in value:
        for i in range(len(d)):
            # if i - 1 >= 0:
            #    s = d[i - 1][0]  # State t-1
            # else:
            #    s = [0.0, 0.0, 0.0, 0.0]
            sample = d[i]

            # s += sample[0]  # State t
            # s.append(sample[1]) # Action
            x.append(np.array(sample[0]) / x_max)  # State t (and scale)
            x_action.append([sample[1]])  # Action
            y.append(np.array(sample[2]) / x_max)  # State t+1 (and scale)
    x, x_action, y = np.array(x), np.array(x_action), np.array(y)
    return x, x_action, y


def preprocess_data_reward_model(data):
    x = []
    y = []
    for d in data:
        s = np.concatenate((d[0], d[1]))
        s = np.append(s, d[2])
        x.append(s)  # State, Next state, Action
        y.append([d[3]])  # Reward
    return np.array(x), np.array(y)


def generate_state_representation(st0, st, a):
    state_representation = st0
    state_representation += st
    state_representation.append(a)
    return np.array(state_representation)


def create_transition_model():
    tflearn.init_graph(seed=1234, num_cores=2, gpu_memory_fraction=0.25)

    net = tflearn.input_data(shape=[None, 5])
    net = tflearn.fully_connected(net, 128, activation='relu')
    net = tflearn.fully_connected(net, 128, activation='relu')
    net = tflearn.fully_connected(net, 128, activation='relu')
    net = tflearn.fully_connected(net, 4, activation='tanh')
    net = tflearn.regression(net, optimizer='Adam', loss='mean_square', restore=False, name="test2")

    return tflearn.DNN(net, tensorboard_dir='/tmp/tflearn_logs/', tensorboard_verbose=3,
                       checkpoint_path='transition_model/checkpoint')


def generate_two_models():
    tflearn.init_graph(seed=1234, num_cores=2, gpu_memory_fraction=0.25)

    net = tflearn.input_data(shape=[None, 9])
    net = tflearn.fully_connected(net, 128, activation='relu')
    net = tflearn.fully_connected(net, 128, activation='relu')
    net = tflearn.fully_connected(net, 128, activation='relu')
    net = tflearn.fully_connected(net, 4, activation='tanh')
    net = tflearn.regression(net, optimizer='Adam', loss='mean_square', name="test2")

    net2 = tflearn.input_data(shape=[None, 9])
    net2 = tflearn.fully_connected(net2, 128, activation='relu')
    net2 = tflearn.fully_connected(net2, 128, activation='relu')
    net2 = tflearn.fully_connected(net2, 1, activation='tanh')
    net2 = tflearn.regression(net2, optimizer='Adam', loss='mean_square', name="test1")

    disnet2 = tflearn.DNN(net2, tensorboard_dir='/tmp/tflearn_logs/', tensorboard_verbose=3,
                          checkpoint_path='reward_model/checkpoint')

    disnet = tflearn.DNN(net, tensorboard_dir='/tmp/tflearn_logs/', tensorboard_verbose=3,
                         checkpoint_path='transition_model/checkpoint')

    return disnet2.load('reward_model/cart_pole_reward_model.tflearn', weights_only=True), disnet.load(
        'transition_model/cart_pole_transition_model.tflearn', weights_only=True)


def train_transition_model(train_data_path='cartpole_data/training.npy', test_data_path='cartpole_data/testing.npy',
                           save=False, file_path='cart_pole_transition_model.tflearn'):
    training_data = np.load(train_data_path)
    X, Y = preprocess_data_transition_model(training_data)

    testing_data = np.load(test_data_path)
    X_test, Y_test = preprocess_data_transition_model(testing_data)

    model = create_transition_model()
    run_id = '2s_3relu_tanh_128_128_128_Adam_mean_square_batch128'
    model.fit(X, Y, n_epoch=10, validation_set=0.1, batch_size=128, show_metric=True, shuffle=True,
              snapshot_epoch=False, run_id=run_id)
    print(model.evaluate(X_test, Y_test, batch_size=128))
    if save:
        model.save(file_path)

    print(run_id)
    print(X[0])
    print(Y[0])
    print(model.predict([X[0]]))

    # 3relu tanh 128 128 128 adam mean_square
    # Accuricy: 88.04%


def create_reward_model():
    tflearn.init_graph(num_cores=2, gpu_memory_fraction=0.25)

    net = tflearn.input_data(shape=[None, 9])
    net = tflearn.fully_connected(net, 128, activation='relu')
    net = tflearn.fully_connected(net, 128, activation='relu')
    net = tflearn.fully_connected(net, 1, activation='tanh')
    net = tflearn.regression(net, optimizer='Adam', loss='mean_square', name="test1", restore=False)

    return tflearn.DNN(net, tensorboard_dir='/tmp/tflearn_logs/', tensorboard_verbose=3)


def train_reward_model(train_data_path='cartpole_data/train_reward.npy', test_data_path='cartpole_data/test_reward.npy',
                       save=False, file_path='cart_pole_reward_model.tflearn'):
    training_data = np.load(train_data_path)
    X, Y = preprocess_data_reward_model(training_data)

    testing_data = np.load(test_data_path)
    X_test, Y_test = preprocess_data_reward_model(testing_data)

    model = create_reward_model()
    run_id = 'test_mean_square_batch128_3'
    model.fit(X, Y, n_epoch=7, validation_set=0.1, batch_size=128, show_metric=True, shuffle=True, run_id=run_id)

    print(model.evaluate(X_test, Y_test, batch_size=128))
    if save:
        model.save(file_path)

    print(run_id)
    print(X[0])
    print(Y[0])
    print(model.predict([X[0]]))
    return model


def gather_reward_data(n_games, save_history=True):
    timesteps = 200
    data = []

    for i in range(n_games):
        env = gym.make('CartPole-v0')
        curr_state = env.reset()

        for time in range(timesteps):
            action = np.random.choice(env.action_space.n)
            # Execute the action in the environment and observe reward
            next_state, reward, done, info = env.step(action)
            if done:
                reward = 0.0
            data.append([curr_state, next_state, action, reward])
            curr_state = next_state
            if done:
                break

    if save_history:
        np_state_history = []
        print("Converting and saving...")
        for i in range(len(data)):
            np_state_history.append(data[i])
        np_state_history = np.array(np_state_history)
        print(np_state_history.shape)
        np.save("cartpole_data/test_reward.npy", np_state_history)
        print("Done saving...")


def create_tf_transition_model():
    # Declare tf graph
    graph = tf.Graph()
    # Build the graph when instantiated
    with graph.as_default():
        tf.set_random_seed(1234)
        W = tf.Variable(tf.random_normal([9, 4]))
        b = tf.Variable(tf.random_normal([4]))

        # Neural Network inputs
        x = tf.placeholder("float", [None, 9])  # Previous state, current state and action taken in current state
        y_ = tf.placeholder(tf.float32, shape=[None, 4])  # Predicted next state

        y = tf.matmul(x, W) + b

        # optim = tf.train.AdamOptimizer().
        init = tf.initialize_all_variables()

    # Declare a TF session and initialize it
    sess = tf.Session(graph=graph)
    sess.run(init)
    return x, y_, y, tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, y_))


class TF_Transition_model:
    def __init__(self, restore=True, save=False):
        self.graph = tf.Graph()
        if restore:
            # Loading model
            self.y_pred, self.X, self.X_action, self.sess = self.tf_transition_model(train=False, save=False,
                                                                                     restore=True)
        else:
            # Training and possibly saving model
            self.y_pred, self.X, self.X_action, self.sess = self.tf_transition_model(train=True, save=save,
                                                                                     restore=False)

    def predict(self, prev_state, curr_state, action):
        state_representation = [curr_state[0], curr_state[1], curr_state[2], curr_state[3]]
        state_representation = np.array(state_representation)
        transition_prediction = self.sess.run(self.y_pred,
                                              feed_dict={self.X: [state_representation], self.X_action: [[action]]})
        # Returning the predicted transition rescaled back to normal
        return transition_prediction[0] * get_min_max()[0]

    def tf_transition_model(self, train_data_path='cartpole_data/training_no_rewards.npy',
                            test_data_path='cartpole_data/testing_no_rewards.npy', train=True, save=False,
                            restore=False,
                            file_path='transition_model/tf_transition_model.ckpt'):
        if (train and restore) or ((not train) and save):
            print("ERROR! No logic in parameters!!")
            return None

        # Load data
        print("Loading data...")
        training_data = np.array([])
        testing_data = np.array([])
        if train:
            training_data = np.load(train_data_path)
            testing_data = np.load(test_data_path)
        X_train, X_train_action, Y_train = preprocess_data_transition_model(training_data)
        X_test, X_test_action, Y_test = preprocess_data_transition_model(testing_data)

        # Parameters
        learning_rate = 0.001
        training_epochs = 20
        batch_size = 128
        display_step = 1
        examples_to_show = 5

        # Network Parameters
        n_input = 4  # Prev state
        n_action = 1  # Action
        n_output = 4  # Predicted next state
        n_hidden_1 = 256  # 1st layer num features
        n_hidden_2 = 256  # 2nd layer num features
        n_hidden_transition = 128  # Transition prediction layer

        w_init_limit = (-0.1, 0.1)

        print("Building graph...")
        with self.graph.as_default():

            # Encode curr_state, add transition prediction with selected action and decode to predicted output state
            X = tf.placeholder("float", [None, n_input])
            X_action = tf.placeholder("float", [None, n_action])
            Y = tf.placeholder("float", [None, n_output])

            weights = {
                'encoder_h1': tf.Variable(
                    tf.random_uniform([n_input, n_hidden_1], minval=w_init_limit[0], maxval=w_init_limit[1])),
                'encoder_h2': tf.Variable(
                    tf.random_uniform([n_hidden_1, n_hidden_2], minval=w_init_limit[0], maxval=w_init_limit[1])),
                'transition_h1': tf.Variable(
                    tf.random_uniform([n_hidden_2 + n_action, n_hidden_transition], minval=w_init_limit[0],
                                      maxval=w_init_limit[1])),
                'decoder_h1': tf.Variable(tf.random_uniform([n_hidden_transition, n_hidden_1], minval=w_init_limit[0],
                                                            maxval=w_init_limit[1])),
                'decoder_h2': tf.Variable(
                    tf.random_uniform([n_hidden_1, n_output], minval=w_init_limit[0], maxval=w_init_limit[1])),
            }
            biases = {
                'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
                'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
                'transition_b1': tf.Variable(tf.random_normal([n_hidden_transition])),
                'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
                'decoder_b2': tf.Variable(tf.random_normal([n_output])),
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
            encoder_op = encoder(X)
            transition_pred = transition(encoder_op, X_action)
            decoder_op = decoder(transition_pred)

            # Prediction
            y_pred = decoder_op
            # Targets (Labels) are the input data.
            y_true = Y

            # Define loss and optimizer, minimize the squared error
            loss_function = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

            # Creates a saver
            saver = tf.train.Saver()

            # Initializing the variables
            init = tf.initialize_all_variables()

        # Launch the graph
        sess = tf.Session(graph=self.graph)
        sess.run(init)
        total_batch = int(len(X_train) / batch_size)
        print("Starting training...")
        print("Total nr of batches:", total_batch)
        if train:
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
                    _, c = sess.run([optimizer, loss_function],
                                    feed_dict={X: batch_xs, X_action: batch_x_action, Y: batch_ys})
                # Display logs per epoch step
                if epoch % display_step == 0 and c is not None:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

        if train:
            print("Finished training!")

        if save:
            save_path = saver.save(sess, file_path)
            print("Model saved in file: %s" % save_path)
        elif restore:
            saver.restore(sess, file_path)
            print("Model restored from file: %s" % file_path)

        if train:
            # Applying encode and decode over test set
            encode_decode = sess.run(y_pred, feed_dict={X: X_test[:examples_to_show],
                                                        X_action: X_test_action[:examples_to_show]})
            print(Y_test[:examples_to_show])
            print(encode_decode)

        return y_pred, X, X_action, sess


if __name__ == '__main__':
    # train_transition_model()
    # train_reward_model(save=False)
    # gather_reward_data(20000)
    # env = gym.make('CartPole-v0')
    # prev_state = env.reset()
    # action = np.random.choice(env.action_space.n)
    # curr_state, reward, done, info = env.step(action)
    # action = np.random.choice(env.action_space.n)
    # real_next_state, reward, done, info = env.step(action)
    model = TF_Transition_model(restore=False, save=True)
    # model.tf_transition_model()

    # print("Real observation:", real_next_state)
    # print("Predicted observation:", model.predict(prev_state, curr_state, action))
