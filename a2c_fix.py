import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.optimizers import Adam
import gym
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from KukaEnv_Cont import KukaVariedObjectEnv # !!! MODIFIED to return feature vector as _step, not image
import os

from reinforce import Reinforce


class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, actor_model, lr, critic_model, critic_lr, n=1, render=False):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.n = n
        self.render = render
        self.lr = lr
        self.critic_lr = critic_lr

        self.actor_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr))
        self.critic_model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.critic_lr))


    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using A2C.
        states, actions, rewards, T = self.generate_episode(env, self.actor_model, self.render)
        V_end = np.zeros(T)
        R_t = np.zeros(T)

        if self.n < T:
            critic_preds_n = self.critic_model.predict(states[self.n:])
            V_end[:T - self.n] = critic_preds_n.flatten()

        #print(rewards)
        for t in reversed(range(T)):
            R_t[t] = V_end[t] + sum([rewards[k + t] if ((t + k) < T) else 0 for k in range(self.n)])

        critic_preds = self.critic_model.predict(states)
        target_vals = R_t - critic_preds.flatten()
        actor_target = np.repeat(np.array([target_vals]), 3, 0).T # get the right dimensions
        critic_target = R_t

        ahistory = self.actor_model.fit(states, actor_target, batch_size=T, verbose=0, epochs=1)
        chistory = self.critic_model.fit(states, critic_target, batch_size=T, verbose=0, epochs=1)

        return ahistory.history['loss'][0], chistory.history['loss'][0], T, np.sum(rewards)



def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=2, help="The value of N in N-step A2C.")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=1, help="Discount Factor")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    render = args.render
    gamma = args.gamma

    # Create the environments
    path = os.path.join(os.getcwd(), "items/")
    print(path)
    env = KukaVariedObjectEnv(path, renders=False) # Continuous!

    # Dx
    actor_model = keras.models.Sequential()
    actor_model.add(Dense(32, input_dim=18, activation='relu'))
    actor_model.add(Dense(32, activation='relu'))
    actor_model.add(Dense(3, activation='linear'))

    # Critic Model
    critic_model = keras.models.Sequential()
    critic_model.add(Dense(32, input_dim=18, activation='relu'))
    critic_model.add(Dense(32, activation='relu'))
    critic_model.add(Dense(32, activation='relu'))
    critic_model.add(Dense(1, activation='linear'))

    agent = A2C(actor_model, lr, critic_model, critic_lr, n, render)

    metrics = {'aloss':[],
               'closs':[],
               'length': [],
               'reward': []}

    for i in range(num_episodes):
        aloss, closs, n, r = agent.train(env, gamma=gamma)

        metrics['aloss'].append(aloss)
        metrics['closs'].append(closs)
        metrics['length'].append(n)
        metrics['reward'].append(r)

        if i % 50 == 0:
            print("On episode {}/{}:".format(i, num_episodes))
        
        if i % 100 == 0 and i > 0: # Print Status
            print("Average rewards for episodes {}-{}: {}".format(i-100, i, np.mean(metrics['reward'][-100:])))

        if i % 500 == 0 and i > 0:
            agent.actor_model.save("a2c_custom/episode_{}_model.h5".format(i))
            print("Checkpoint: Saved Model at Episode {}".format(i))
            with open("a2c_custom/metrics_" + str(i) + '.pkl', 'wb') as f:
                pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)
        

    with open(str(num_episodes) + "_" + str(gamma) + 'metrics_tot1' + '.pkl', 'wb') as f:
        pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main(sys.argv)
