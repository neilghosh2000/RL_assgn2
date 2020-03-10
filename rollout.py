#!/usr/bin/env python

import click
import numpy as np
import gym
from gym.envs.registration import register


def chakra_get_action(theta, ob, rng=np.random):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    return rng.normal(loc=mean, scale=1.)		 


def get_log_policy(x, y):
    grad_log = np.exp(-((x-y)*(x-y)) / 2)
    grad_log /= np.sqrt(2*np.pi)
    return np.log(grad_log)


def get_policy_gradient(x, y):
    grad = []
    for i in range(len(x)):
        l = []
        for j in range(len(y)):
            l.append(x[i]*y[j])
        grad.append(l)
    return np.array(grad)


def include_bias(x):
    biased_x = list()
    biased_x.append(1)
    for i in range(len(x)):
        biased_x.append(x[i])
    return np.array(biased_x)


@click.command()
@click.argument("env_id", type=str, default="chakra")
def main(env_id):
    # Register the environment
    register(id='chakra-v0',
             entry_point='PolicyGradient.policy_gradient:Chakra')
    rng = np.random.RandomState(42)
    if env_id == 'chakra':
        env = gym.make('chakra-v0')
        get_action = chakra_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError(
            "Unsupported environment: must be 'chakra' ")

    env.seed(42)

    # Initialize parameters
    theta = rng.normal(scale=0.01, size=(action_dim, obs_dim + 1))
    for itr in range(2000):
        trajectory_count = 50
        trajectory_states = []
        trajectory_rewards = []
        for eps in range(trajectory_count):
            gamma = 1
            ob = env.reset()
            done = False
            # Only render the first trajectory
            # Collect a new trajectory
            rewards = []
            states = []
            actions = []
            discounted_reward = 0
            while not done:
                action = get_action(theta, ob, rng=rng)
                actions.append(action)
                next_ob, rew, done, _ = env.step(action)
                ob = next_ob
                if ob[0] < -1 or ob[0] > 1 or ob[1] < -1 or ob[1] > 1:
                    env.reset()
                # env.render(mode='human')
                rewards.append(rew)
                discounted_reward += gamma*rew
                gamma *= 0.9
                states.append(ob)
            grad = np.zeros(theta.shape)
            for j in range(len(states)):
                action = actions[j]
                state = states[j]
                reward = rewards[j]
                mean = theta.dot(include_bias(state))
                log_policy = get_log_policy(action, mean)
                grad_policy = get_policy_gradient(log_policy, include_bias(state))
                grad_policy = grad_policy / (np.linalg.norm(grad_policy))
                grad += grad_policy
                print(grad)
                discounted_reward -= reward
                discounted_reward /= 0.9

            trajectory_states.append(states)
            trajectory_rewards.append(rewards)
            # print("Episode reward: %.2f" % np.sum(rewards))


if __name__ == "__main__":
    main()
