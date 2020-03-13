#!/usr/bin/env python

import click
import numpy as np
import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt


def chakra_get_action(theta, ob, rng=np.random):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    action = rng.normal(loc=mean, scale=0.5)
    return action


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
    rng = np.random.RandomState(42)
    if env_id == 'chakra':
        register(id='chakra-v0',
                 entry_point='PolicyGradient.policy_gradient:Chakra')
        env = gym.make('chakra-v0')
        get_action = chakra_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    elif env_id == 'vishamC':
        register(id='vishamC-v0',
                 entry_point='PolicyGradient.policy_gradient:VishamC')
        env = gym.make('vishamC-v0')
        get_action = chakra_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError(
            "Unsupported environment: must be 'chakra' ")

    env.seed(42)

    reward_list = []
    steps_list = []
    # Initialize parameters
    theta = rng.normal(scale=0.01, size=(action_dim, obs_dim + 1))
    print(theta)
    for itr in range(750):
        trajectory_count = 50
        total_reward = 0
        total_steps = 0
        trajectory_states = []
        trajectory_rewards = []
        grad = np.zeros(theta.shape)
        for eps in range(trajectory_count):
            print(itr, eps)
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
                states.append(ob)
                if abs(ob[0]) > 1 or abs(ob[1]) > 1:
                    ob = env.reset()
                    rew = -5
                rewards.append(rew)
                # env.render(mode='human')
                discounted_reward += gamma*rew
                gamma *= 0.9
            curr_gamma = 1
            total_reward += sum(rewards)
            total_steps += len(rewards)
            for j in range(len(states)):
                action = actions[j]
                state = states[j]
                reward = rewards[j]
                mean = theta.dot(include_bias(state))
                grad_policy = get_policy_gradient(action - mean, include_bias(state))
                grad_policy = grad_policy / (np.linalg.norm(grad_policy) + 1e-8)
                grad_policy = grad_policy*0.0001*curr_gamma*discounted_reward
                # theta += grad_policy
                # grad += grad_policy
                grad += grad_policy
                curr_gamma *= 0.9
                discounted_reward -= reward
                discounted_reward /= 0.9
            trajectory_states.append(states)
            trajectory_rewards.append(rewards)
        theta += grad / trajectory_count
        reward_list.append(total_reward / trajectory_count)
        steps_list.append(total_steps / trajectory_count)
    fig_rewards = plt.figure().add_subplot(111)
    fig_steps = plt.figure().add_subplot(111)
    fig_rewards.set_xlabel('Iterations')
    fig_rewards.set_ylabel('Average Reward')
    fig_rewards.plot(range(len(reward_list)), reward_list)
    print(theta)

    fig_steps.set_xlabel('Iterations')
    fig_steps.set_ylabel('Average Episode Length')
    fig_steps.plot(range(len(steps_list)), steps_list)
    plt.show()


if __name__ == "__main__":
    main()
