import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.optimizers import Adam
import pickle
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr, render=False):
        self.model = model
        self.render = render
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr))

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        states, actions, rewards, n = self.generate_episode(env, self.critic_model, self.render)

        G = np.cumsum(rewards[::-1])[::-1] # gamma = 1

        #for t in reversed(range(n)):
            #G[t] = sum([gamma ** (k - t) * rewards[k] for k in range(t, n)])

        # One-Hot Encoding of G
        y_true = (np.tile(G, (4,1)).T * keras.utils.to_categorical(actions, num_classes = 4)) / 100

        history = self.model.fit(x = states, y = y_true, batch_size = n,
                                                epochs = 1, verbose = 0)

        return history.history['loss'][0], n, np.sum(rewards)

    def generate_episode(self, env, actor_model, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # - an int of episode length
        if render:
            env = gym.wrappers.Monitor(env, '/tmp/reinforce_v1/')

        states = []
        actions = []
        rewards = []

        state = env.reset()
        done = False

        while not done:
            action_probabilities = actor_model.predict(state.reshape((1, 18)))
            action = action_probabilities.flatten()

            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        return np.array(states), np.array(actions), np.array(rewards), len(states)


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=0.0025, help="The learning rate.")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=1.0, help="The discount factor.")

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
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    gamma = args.gamma
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # Load the policy model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # TODO: Train the model using REINFORCE and plot the learning curve.    
    agent = Reinforce(model, lr, render)

    metrics = {'loss':[],
               'length': [],
               'reward': [],
               'reward_mean': [],
               'reward_std': []}

    for i in range(num_episodes):
        loss, n, r = agent.train(env, gamma=gamma)

        metrics['loss'].append(loss)
        metrics['length'].append(n)
        metrics['reward'].append(r)

        if i % 50 == 0:
            print("On episode {}/{}:".format(i, num_episodes))
        
        if i % 250 == 0:
            rewards = np.zeros((100,2))
            for e in range(100):
                _, _, r, n = agent.generate_episode(env)
                rewards[e] = [sum(r), n]

            metrics['reward_mean'].append(rewards[:,0].mean())
            metrics['reward_std'].append(rewards[:,0].std())

            print()
            print("Avg. Rewards: {}".format(round(metrics['reward_mean'][-1],2)), end=' +- ')
            print(round(metrics['reward_std'][-1],2))
            print("Avg. Episode Length: {}".format(round(rewards[:,1].mean(),2)))
            print("Best Reward So Far: {}".format(round(max(metrics['reward']),2)), end='\n\n')

        if i % 500 == 0 and i > 0:
            agent.model.save("mdl/episode_{}_model.h5".format(i))
            print("Checkpoint: Saved Model at Episode {}".format(i))

        if np.mean(metrics['reward'][-100:]) > 200:
            print("SOLVED: STOPPING EARLY AT EPISODE {}".format(i))
            agent.model.save("mdl/episode_{}_model.h5".format(i))
            break

    with open(str(num_episodes) + "_" + str(gamma) + 'metrics' + '.pkl', 'wb') as f:
        pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main(sys.argv)