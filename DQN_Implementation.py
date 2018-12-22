#!/usr/bin/env python
import tensorflow as tf, keras, numpy as np, gym, sys, copy, argparse, random, math
from keras.layers import Input, Dense, Add, Lambda
from keras.models import Model
from keras.optimizers import Adam

class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, input_dim, output_dim, learning_rate):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.

        # Define the model in terms of environment information
        self.observations = Input(shape=(input_dim,))
        self.hidden = Dense(16, activation='relu')(self.observations)
        self.hidden = Dense(16, activation='relu')(self.hidden)
        self.hidden = Dense(8, activation='relu')(self.hidden)
        self.actions = Dense(output_dim, activation='linear')(self.hidden)

        # Compile the model
        self.model = Model(inputs=self.observations, outputs=self.actions)
        self.model.compile(optimizer=Adam(lr=learning_rate), loss='mse')

    def save_model(self, filename):
        keras.models.save_model(self.model, filename)

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights.
        self.model.save_weights(suffix)

    def load_model(self, model_file):
        # Helper function to load an existing model.
        self.model = keras.models.load_model(model_file)

    def load_model_weights(self,weight_file):
        # Helper funciton to load model weights.
        pass

class Dueling_QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, input_dim, output_dim, learning_rate):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.

        # Define the model in terms of environment information
        self.observations = Input(shape=(input_dim,))
        self.hidden = Dense(16, activation='relu')(self.observations)
        self.hidden = Dense(16, activation='relu')(self.hidden)

        # Value Network
        self.value = Dense(8, activation='relu')(self.hidden)
        self.value = Dense(1, activation='linear')(self.value)

        # Advantage Network
        self.advantage = Dense(8, activation='relu')(self.hidden)
        self.advantage = Dense(output_dim, activation='linear')(self.advantage)

        self.advantage_portion = Lambda(lambda a: a - tf.reduce_mean(
                                a, axis=-1, keep_dims=True))(self.advantage)
        self.value_portion = Lambda(lambda v: tf.tile(v, [1, output_dim]))(self.value)

        # Q(s,a) = V(s,a) + (A(s,a) - mean(A(s,a)))
        # This is equation (9) in [4]
        self.actions = Add()([self.value_portion, self.advantage_portion])

        # Compile the model
        self.model = Model(inputs=self.observations, outputs=self.actions)
        self.model.compile(optimizer=Adam(lr=learning_rate), loss='mse')

    def save_model(self, filename):
        keras.models.save_model(self.model, filename)

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights.
        self.model.save_weights(suffix)

    def load_model(self, model_file):
        # Helper function to load an existing model.
        self.model = keras.models.load_model(model_file)

    def load_model_weights(self,weight_file):
        # Helper funciton to load model weights.
        pass

class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.
        self.experience = list()
        self.max_memory_size = memory_size
        self.burn_in_size = burn_in

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        # Returns a numpy array of the 5-tuples
        return np.array(random.sample(self.experience, batch_size))

    def append(self, transition):
        # Appends transition to the memory.
        self.experience.append(transition)

        if len(self.experience) > self.max_memory_size:
            self.experience.pop(0) # remove the first (least recent) transition

class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, dueling=False, double=False, learning_rate=0.001,
                 training_episodes=10000, epsilon=0.5, batch_size=32, gamma=0.99,
                 target_gap=100, render=False, debug=False, render_folder=None,
                 burn_in_size=10000, memory_size=50000):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.

        # Environmental Attributes
        self.env_name = environment_name
        self.Memory = Replay_Memory(memory_size, burn_in_size) # default values
        self.environment = gym.make(environment_name)
        self.test_environment = gym.make(environment_name)
        self.input_dim = self.environment.observation_space.shape[0]
        self.output_dim = self.environment.action_space.n
        if dueling:
            self.Model = Dueling_QNetwork(self.input_dim, self.output_dim,
                                  learning_rate)
            self.Target = Dueling_QNetwork(self.input_dim, self.output_dim,
                                   learning_rate)
        else:
            self.Model = QNetwork(self.input_dim, self.output_dim, learning_rate)
            self.Target = QNetwork(self.input_dim, self.output_dim, learning_rate)

        self.render = render
        self.render_folder = render_folder
        self.debug = debug
        self.double = double

        # Hyperparameters
        self.epsilon = epsilon
        self.training_episodes = training_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.target_gap = target_gap
        self.epsilon_anneal = (self.epsilon - 0.05) / self.training_episodes


    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.random() < self.epsilon:
            # randomly sample from action space
            return self.environment.action_space.sample()

        else:
            # return the best action
            return np.argmax(q_values)

    def small_epislon_greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        if np.random.random() < 0.05:
            # randomly sample from action space
            return self.environment.action_space.sample()

        else:
            # return the best action
            return np.argmax(q_values)

    def two_step_lookahead(self, state_info):
        # Loop through every action
        best_action = -1
        best_q = -10000000000
        for i in range(self.output_dim):
            next_state = CartPole_Transition_fn(state_info, i)
            new_q = np.max(self.Model.model.predict(next_state.reshape((1, self.input_dim))))

            if new_q > best_q:
                best_q = new_q
                best_action = i

        return best_action

    def train(self):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        episode_rewards = []
        performance = []
        look_ahead = []
        if self.render:
            self.environment = gym.wrappers.Monitor(self.environment, self.render_folder,
            video_callable=lambda x: x % (int(self.training_episodes / 6))  == 0 or x == self.training_episodes)

        for i in range(self.training_episodes):
            # The first state isn't terminal
            state = self.environment.reset()
            done = False

            # CartPole 2-step lookahead
            if i % 100 == 0 and self.env_name == 'CartPole-v0':
                look_ahead.append(self.test(test_size=20, look_ahead=True))

            # Get performance and Checkpoint weights
            if i % 100 == 0:
                performance.append(self.test(test_size=20))
                self.Model.model.save_weights(self.render_folder + '/weights{}.h5'.format(i))

            # Ever target_gap Episodes, update the target model to the current model weights
            if i % self.target_gap == 0:
                self.Target.model.set_weights(self.Model.model.get_weights())

            # Complete the episode
            episode_reward = 0
            while not done:
                q_values = self.Model.model.predict(state.reshape((1, self.input_dim)), batch_size=1)
                action = self.epsilon_greedy_policy(q_values)

                # Take another epsilon-greedy step
                next_state, reward, done, _ = self.environment.step(action)
                episode_reward += reward

                # Add to experience and get a minibatch
                self.Memory.append((state, action, reward, next_state, done))
                state = next_state # move on

                if done:
                    break

                # Update the Model from memory
                if self.debug:
                    debug = (abs(episode_reward) == 1 and i % 500 == 0)
                else:
                    debug = False
                loss = self.batch_update(debug=debug)
                self.print_status(loss, episode_reward, i)

            episode_rewards.append(episode_reward)
            self.epsilon -= self.epsilon_anneal

        return np.array(episode_rewards), np.array(performance), np.array(look_ahead)

    def test(self, test_size=100, model_file=None, look_ahead=False):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        rewards = 0
        if model_file:
            self.Model.model = keras.models.load_model(model_file)

        for i in range(test_size):
            state = self.test_environment.reset()
            done = False

            while not done:
                q_values = self.Model.model.predict(
                    state.reshape((1, self.input_dim)), batch_size=1)

                if look_ahead:
                    action = self.two_step_lookahead(state)
                else:
                    action = self.small_epislon_greedy_policy(q_values)

                # Take another epsilon-greedy step
                next_state, reward, done, _ = self.test_environment.step(action)
                rewards += reward

                state = next_state

        return rewards / test_size

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        state = self.environment.reset() # reset to initialize

        for i in range(self.Memory.burn_in_size):
            action = self.environment.action_space.sample() # choose a random action
            next_state, reward, done, _ = self.environment.step(action) # transition

            # append the 5-tuple to memory
            self.Memory.append((state, action, reward, next_state, done))

            if done:
                state = self.environment.reset()
            else:
                state = next_state # if we're not done, move on

    def batch_update(self, debug=False):
        minibatch = self.Memory.sample_batch(self.batch_size)

        # Generate Y values from minibatch
        states = np.stack(minibatch[:,0])
        actions = minibatch[:,1].astype(int)
        rewards = minibatch[:,2]
        next_states = np.stack(minibatch[:,3])
        is_done = minibatch[:,4]
        inds = range(self.batch_size)
        online_qs = self.Model.model.predict(next_states)

        if self.double:
            online_actions = np.argmax(online_qs, axis=1)
            newrewards = self.Target.model.predict(next_states)[inds, online_actions]
        else:
            newrewards = np.max(online_qs, axis=1)

        y = self.Target.model.predict(states)

        y[inds, actions] = rewards + (1 - is_done) * self.gamma * newrewards

        if debug:
            print(np.round(y[:10], 2))

        history = self.Model.model.fit(states, y, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        return loss

    def print_status(self, loss, rewards, episode, episode_gap = 100, reward_gap = 10):
        if episode % episode_gap == 0:
            if rewards % reward_gap == 0:
                print("Episode: {}, Rewards: {},  Loss: {}, Epsilon: {}"\
                      .format(episode, rewards, round(loss, 2), round(self.epsilon, 2)))


# Transition function taken from OpenAI Gym source code!
def CartPole_Transition_fn(state_info, action):
    [x, x_dot, theta, theta_dot] = state_info
    force = 10.0 if action == 1 else -10.0
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    temp = (
           force + 0.6 * theta_dot * theta_dot * sintheta) / 1.1
    thetaacc = (9.8 * sintheta - costheta * temp) / (0.5 * (
    4.0 / 3.0 - 0.1 * costheta * costheta / 1.1))
    xacc = temp - 0.6 * thetaacc * costheta / 1.1

    newx = x + 0.02 * x_dot
    newx_dot = x_dot + 0.02 * xacc
    newtheta = theta + 0.02 * theta_dot
    newtheta_dot = theta_dot + 0.02 * thetaacc

    return np.array([newx, newx_dot, newtheta, newtheta_dot])

from matplotlib import pyplot as plt
import seaborn as sns

def train_and_save(agent, modelpath, title):
    agent.burn_in_memory()
    training, performance, lookahead = agent.train()
    agent.Model.save_model(modelpath + 'model.h5')
    agent.Model.model.save_weights(modelpath + 'weights.h5')
    np.save(modelpath + "rewards.npy", training)
    np.save(modelpath + "performance.npy", performance)

    inds = list(range(1, agent.training_episodes + 1, 100))
    avg_r = np.mean(training.reshape((-1, 100)), axis=1)

    legend = ['Training Curve', 'Performance Plot']
    plt.plot(inds, avg_r)
    plt.plot(inds, performance)
    if len(lookahead) > 1:
        legend.append('Two-Step-Lookahead Peformance')
        plt.plot(inds, lookahead)

    plt.title(title)
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Rewards')
    plt.legend(legend)
    plt.savefig(modelpath + 'pplot.jpg')
    plt.close()

def generate_plots(modelpath, title):
    training = np.load(modelpath + "rewards.npy")
    performance = np.load(modelpath + "performance.npy")

    inds = list(range(1, 10001, 100))
    avg_r = np.mean(training.reshape((-1, 100)), axis=1)

    legend = ['Training Curve', 'Performance Plot']
    plt.plot(inds, avg_r)
    plt.plot(inds, performance)

    plt.title(title)
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Rewards')
    plt.legend(legend)
    plt.savefig(modelpath + 'pplot.jpg')
    plt.close()

agent = DQN_Agent('CartPole-v0',
                  dueling=True)

lookahead = []

for i in range(0,9901,100):
    try:
        agent.Model.model.load_weights('Dueling/CartPole3/videos/weights{}.h5'.format(i))
        lookahead.append(agent.test(20, look_ahead=True))
    except:
        print('couldnt load weights {}'.format(i))

lookahead.append(200)

np.save('Dueling/Cartpole3/lookahead.npy', np.array(lookahead))
