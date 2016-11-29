'''
Created on Jun 26, 2016
@author: Davide Nitti
https://github.com/davidenitti/ML/tree/master/RL/gymeval
'''

import gym
import logging
import numpy as np
import math, time
import tensorflow as tf
import pickle
from gym import spaces
import os.path

import model_based_learner_lunar_2 as model_based_learner


def multilayer_perceptron(_X, numhidden, regulariz, minout=None, maxout=None, initbias=0., outhidden=-1, seed=None):
    if seed is not None:
        tf.set_random_seed(seed)
    numlayers = len(numhidden)
    layer_i = _X
    regul = 0.
    for i in range(1, numlayers - 1):
        w = tf.Variable(
            tf.truncated_normal([numhidden[i - 1], numhidden[i]], stddev=1. / math.sqrt(float(numhidden[i - 1]))),
            name="w" + str(i))
        b = tf.Variable(tf.zeros([numhidden[i]]), name="b" + str(i))
        layer_i = tf.nn.tanh(tf.add(tf.matmul(layer_i, w), b))
        regul += tf.nn.l2_loss(w) * regulariz[i - 1]  # + tf.nn.l2_loss(b)*initbias
        if outhidden == i:
            hidlayer = layer_i
        print('w', w.get_shape(), 'b', b.get_shape(), 'l', layer_i.get_shape(), regulariz[i - 1])
    w = tf.Variable(tf.truncated_normal([numhidden[numlayers - 2], numhidden[numlayers - 1]],
                                        stddev=1. / math.sqrt(float(numhidden[numlayers - 2]))),
                    name="w" + str(numlayers - 1))
    b = tf.Variable(tf.zeros([numhidden[numlayers - 1]]), name="b" + str(numlayers - 1))

    if minout is None:
        layer_out = tf.matmul(layer_i, w) + b + initbias
    else:
        layer_out = tf.nn.sigmoid(tf.matmul(layer_i, w) + b + initbias) * (maxout - minout) + minout
    print(regulariz, numlayers, len(numhidden))
    print('w', w.get_shape(), 'b', b.get_shape(), 'l', layer_out.get_shape(), regulariz[numlayers - 2])
    regul += tf.nn.l2_loss(w) * regulariz[numlayers - 2]  # +tf.nn.l2_loss(b)*initbias
    if outhidden >= 0:
        return layer_out, regul, hidlayer
    else:
        return layer_out, regul

class deepQAgent(object):
    def __del__(self):
        print('deepQAgent died')
        self.close()

    def save(self, filename=None):
        if filename is None:
            filename = self.config['file']
        with open(filename + ".p", "wb") as input_file:
            pickle.dump((self.observation_space, self.action_space, self.reward_range, self.config), input_file)
        save_path = self.saver.save(self.sess, filename + ".tf")
        print("Saved in:", save_path)

    def __init__(self, env, observation_space, action_space, reward_range, **userconfig):
        if userconfig["file"] is not None and os.path.isfile(userconfig["file"] + ".p"):
            with open(userconfig["file"] + ".p", "rb") as input_file:
                self.observation_space, self.action_space, self.reward_range, self.config = pickle.load(input_file)
                # overwrite some parameter
                self.config["initial_learnrate"] = userconfig["initial_learnrate"]
                self.config["eps"] = userconfig["eps"]
                self.config["batch_size"] = userconfig["batch_size"]
                self.config["probupdate"] = userconfig["probupdate"]
                self.config["lambda"] = userconfig["lambda"]
                self.config["momentum"] = userconfig["momentum"]
                self.config["memsize"] = userconfig["memsize"]
                if not ("featureset" in self.config):
                    self.config["featureset"] = userconfig["featureset"]
        else:
            self.observation_space = observation_space
            self.action_space = action_space
            self.env = env
            self.reward_range = reward_range
            self.config = {
                "memsize": 50000,
                "scalereward": 1.,
                "probupdate": 0.25,
                "lambda": 0.1,
                "past": 0,
                "eps": 0.15,  # Epsilon in epsilon greedy policies
                "decay": 0.996,  # Epsilon decay in epsilon greedy policies
                "initial_learnrate": 0.008,
                "decay_learnrate": 0.999,
                "discount": 0.99,
                "batch_size": 75,
                "hiddenlayers": [300],
                "regularization": [0.0001, 0.000001],
                "momentum": 0.1,
                "file": None,
                "seed": None}
            self.config.update(userconfig)

        if self.config["seed"] is not None:
            np.random.seed(self.config["seed"])
            print("seed", self.config["seed"])
        self.isdiscrete = isinstance(self.action_space, gym.spaces.Discrete)
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise Exception('Observation space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(
                observation_space, self))

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.learnrate = tf.train.exponential_decay(self.config['initial_learnrate'], self.global_step, 100,
                                                    self.config['decay_learnrate'], staircase=True, name="learnrate")
        self.bias = tf.Variable(0.0, trainable=False, name="bias")

        self.initQnetwork()

    def scaleobs(self, obs):
        if np.isinf(self.observation_space.low).any() or np.isinf(self.observation_space.high).any():
            return obs
        else:  # from -3 to 3
            o = (obs - self.observation_space.low) / (
                self.observation_space.high - self.observation_space.low) * 2. - 1.
            return o * 3

    def epsilon(self, episode=None):
        if episode is None:
            return 0.
        else:
            return self.config['eps'] * self.config['decay'] ** episode

    def incbias(self, l):
        self.sess.run(tf.assign(self.bias, self.bias + l))

    def getbias(self):
        return self.sess.run(self.bias)

    def getlearnrate(self):
        return self.sess.run(self.learnrate)

    def stateautoencoder(self, n_input):
        self.plotautoenc = False
        self.outstate, regul, self.hiddencode = multilayer_perceptron(self.sa, [n_input, 2, n_input],
                                                                      [0.000001, 0.000001], outhidden=1,
                                                                      seed=self.config["seed"])
        # print self.x.get_shape(),self.outstate.get_shape()
        self.costautoenc = tf.reduce_mean((self.sa - self.outstate) ** 2) + regul
        self.optimizerautoenc = tf.train.RMSPropOptimizer(self.learnrate, 0.9, 0.05).minimize(self.costautoenc,
                                                                                              global_step=self.global_step)

    def initQnetwork(self):
        if self.isdiscrete:
            n_input = self.observation_space.shape[0] * (self.config['past'] + 1)
            self.n_out = self.action_space.n

        self.x = tf.placeholder("float", [None, n_input], name="self.x")
        self.y = tf.placeholder("float", [None, 1], name="self.y")
        self.yR = tf.placeholder("float", [None, 1], name="self.yR")

        print('obs', n_input, 'action', self.n_out)
        self.Qrange = (self.reward_range[0] * 1. / (1. - self.config['discount']),
                       self.reward_range[1] * 1. / (1. - self.config['discount']))
        print(self.Qrange)

        self.Q, regul = multilayer_perceptron(self.x, [n_input] + self.config['hiddenlayers'] + [self.n_out],
                                              self.config['regularization'],
                                              initbias=.0, seed=self.config["seed"])  # ,self.Qrange[0],self.Qrange[1])

        self.R, regulR = multilayer_perceptron(self.x, [n_input, 100, self.n_out], self.config['regularization'],
                                               initbias=.0, seed=self.config["seed"])  # ,self.Qrange[0],self.Qrange[1])

        if self.isdiscrete:
            self.curraction = tf.placeholder("float", [None, self.n_out], name="curraction")

            self.singleQ = tf.reduce_sum(self.curraction * self.Q,
                                         reduction_indices=1)  # tf.reduce_sum(input_tensor, reduction_indices, keep_dims, name)#tf.slice(self.Q, [0,self.curraction],[-1,1])  #self.Q[:,self.out]
            self.singleQ = tf.reshape(self.singleQ, [-1, 1])

            self.singleR = tf.reduce_sum(self.curraction * self.R,
                                         reduction_indices=1)  # tf.reduce_sum(input_tensor, reduction_indices, keep_dims, name)#tf.slice(self.Q, [0,self.curraction],[-1,1])  #self.Q[:,self.out]
            self.singleR = tf.reshape(self.singleR, [-1, 1])

            self.errorlistR = (self.singleR - self.yR) ** 2
            self.errorlist = (self.singleQ - self.y) ** 2

        self.cost = tf.reduce_mean(self.errorlist) + regul
        self.costR = tf.reduce_mean(self.errorlistR) + regulR
        self.lastcost = 0.
        self.optimizer = tf.train.RMSPropOptimizer(self.learnrate, 0.9, self.config['momentum']).minimize(self.cost,
                                                                                                          global_step=self.global_step)

        self.optimizerR = tf.train.RMSPropOptimizer(self.learnrate, 0.9, 0.).minimize(self.costR,
                                                                                      global_step=self.global_step)

        self.sa = tf.placeholder("float", [None, n_input + self.n_out], name="sa")
        self.stateautoencoder(n_input + self.n_out)

        self.saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.memory = []
        self.errmemory = []
        if self.config['file'] is None or (not os.path.isfile(self.config['file'] + ".tf")):
            self.sess.run(tf.initialize_all_variables())
        else:
            print("loading " + self.config['file'] + ".tf")
            self.saver.restore(self.sess, self.config['file'] + ".tf")
        self.sess.run(tf.assign(self.global_step, 0))


    def learn(self, state, action, obnew, reward, notdone, nextaction):
        if self.isdiscrete:
            cost = 0
            target = reward + self.config['discount'] * self.maxq(obnew) * notdone

            target = target.reshape(1, )
            allstate = state.reshape(1, -1)
            allaction = np.array([action])
            alltarget = target
            indexes = [-1]
            update = (np.random.random() < self.config['probupdate'])
            if update:
                if len(self.memory) > self.config['batch_size']:
                    ind = np.random.choice(len(self.memory), self.config['batch_size'])
                    for j in ind:
                        s, a, r, onew, d, Q, nextstate = self.memory[j]

                        if Q is not None:
                            alternativetarget = Q
                        else:
                            if self.config['lambda'] > 0.:
                                limitd = 1000
                                alternativetarget = r
                                gamma = self.config['discount']
                                offset = 0
                                if nextstate is None:
                                    alternativetarget += gamma * self.maxq(onew) * d

                                while nextstate is not None and offset < limitd:
                                    offset += nextstate
                                    n = j + offset

                                    alternativetarget += gamma * self.memory[n][2]
                                    gamma = gamma * self.config['discount']
                                    if self.memory[n][6] is None or not (offset < limitd):
                                        alternativetarget += gamma * self.maxq(self.memory[n][3]) * self.memory[n][4]
                                    nextstate = self.memory[n][6]
                            else:
                                alternativetarget = 0.
                            self.memory[j][5] = alternativetarget

                        alternativetarget = alternativetarget * self.config['lambda'] + (r + self.config[
                            'discount'] * self.maxq(onew) * d) * (1. - self.config['lambda'])

                        alltarget = np.concatenate((alltarget, alternativetarget), 0)

                        allstate = np.concatenate((allstate, s.reshape(1, -1)), 0)
                        allaction = np.concatenate((allaction, np.array([a])), 0)
                        indexes.append(j)
                    allactionsparse = np.zeros((allstate.shape[0], self.n_out))
                    allactionsparse[np.arange(allaction.shape[0]), allaction] = 1.

                    self.sess.run(self.optimizer, feed_dict={self.x: allstate, self.y: alltarget.reshape((-1, 1)),
                                                             self.curraction: allactionsparse})

                    if self.plotautoenc:
                        sa = np.concatenate((allstate, allactionsparse), 1)
                        self.sess.run(self.optimizerautoenc, feed_dict={self.sa: sa, self.outstate: sa})
            allactionsparse = np.zeros((allstate.shape[0], self.n_out))
            allactionsparse[np.arange(allaction.shape[0]), allaction] = 1.
            indexes = np.array(indexes)
            erlist = self.sess.run(self.errorlist, feed_dict={self.x: allstate, self.y: alltarget.reshape((-1, 1)),
                                                              self.curraction: allactionsparse})
            erlist = erlist.reshape(-1, )
            self.errmemory.append(erlist[0])

            for i, er in enumerate(erlist):
                self.errmemory[indexes[i]] = er
            self.errmemory = self.errmemory[-self.config['memsize']:]

            if len(self.memory) > 0 and np.array_equal(self.memory[-1][3], state):
                self.memory[-1][6] = 1
            self.memory.append([state, action, reward, obnew, notdone, None, None])

            self.memory = self.memory[-self.config['memsize']:]
            return cost

    def maxq(self, observation):
        if self.isdiscrete:
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            return np.max(self.sess.run(self.Q, feed_dict={self.x: observation})).reshape(1, )

    def argmaxq(self, observation):
        if self.isdiscrete:
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            return np.argmax(self.sess.run(self.Q, feed_dict={self.x: observation}))


    def act(self, observation, episode=None):
        eps = self.epsilon(episode)

        # epsilon greedy.
        if np.random.random() > eps:
            action = self.argmaxq(observation)  # self.softmaxq(observation)#
        else:
            action = self.action_space.sample()
        return action

    def close(self):
        self.sess.close()


def gather_random_data(env, n_steps=1000):
    total_steps = 0
    experience = []
    while True:
        state = env.reset()
        for time_step in range(env.spec.timestep_limit):
            action = np.random.choice(env.action_space.n)
            next_state, reward, done, info = env.step(action)
            experience.append([state, action, next_state, reward, done])
            state = next_state
            total_steps += 1
            if done or total_steps >= n_steps:
                break
        if total_steps >= n_steps:
            break
    return experience


def do_imagination_rollouts(agent, env, episodes, num_steps=None):
    if episodes > 0:
        # Gathering data for imagination rollouts
        gathered_data_size = 50000
        experience = gather_random_data(env, gathered_data_size)

        # Train transition model on gathered data and pre-train actor-critic from imagination rollouts
        training_data = np.array(experience)
        print("Training size:", len(training_data))
        test_data_r = np.load('lunarlander_data_done/random_agent/testing_data.npy')
        # test_data_ac = np.load('lunarlander_data/actor_critic/testing_data.npy')
        transition_model = model_based_learner.TF_Transition_model(env, w_init_limit=(-0.2, 0.2), display_step=100)
        acc1, acc2 = transition_model.train(training_epochs=200, learning_rate=0.0005, batch_size=512,
                                            training_data=training_data, test_data_r=test_data_r,
                                            test_data_ac=None, logger=False)
        # Doing imagination episodes if test error is less than a threshold
        if num_steps is None:
            num_steps = 100  #env.spec.timestep_limit # Max the num steps because the imagination rollouts is inaccurate

        for episode in range(episodes):
            total_rew = 0.
            cost = 0.
            ob = env.reset()
            ob = agent.scaleobs(ob)
            ob1 = np.copy(ob)

            for t in range(num_steps):
                a = agent.act(ob1, episode)
                obnew, reward, done, _ = transition_model.predict(ob1, a)
                obnew = agent.scaleobs(obnew)
                reward *= agent.config['scalereward']
                obnew1 = np.concatenate((ob1[ob.shape[0]:], obnew))

                cost += agent.learn(ob1, a, obnew1, reward, 1. - 1. * done, agent.act(obnew1, episode))

                ob1 = obnew1
                total_rew += reward
                if done:
                    break
            print("Episode:", episode, "Reward:", total_rew, "Timesteps:", t + 1, "Cost:", cost)


def do_rollout(agent, env, episode, num_steps=None, render=False):
    if num_steps is None:
        num_steps = env.spec.timestep_limit
    total_rew = 0.
    cost = 0.

    ob = env.reset()
    ob = agent.scaleobs(ob)
    ob1 = np.copy(ob)
    for _ in range(agent.config["past"]):
        ob1 = np.concatenate((ob1, ob))
    listob = [ob1]
    listact = []
    for t in range(num_steps):
        a = agent.act(ob1, episode)
        (obnew, reward, done, _info) = env.step(a)
        obnew = agent.scaleobs(obnew)
        reward *= agent.config['scalereward']
        listact.append(a)
        obnew1 = np.concatenate((ob1[ob.shape[0]:], obnew))

        listob.append(obnew1)

        start = time.time()
        cost += agent.learn(ob1, a, obnew1, reward, 1. - 1. * done, agent.act(obnew1, episode))

        if render and (t % 20 == 0 or done):
            print('learn time', (time.time() - start) * 100., agent.maxq(ob1), reward)
        ob1 = obnew1
        total_rew += reward
        if render and t % 2 == 0:
            env.render()
        if done:
            break
    return total_rew, t + 1, cost, listob, listact
