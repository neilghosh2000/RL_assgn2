#!/usr/bin/env python

import click
import numpy as np
import gym


def chakra_get_action(theta, ob, rng=np.random):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    return rng.normal(loc=mean, scale=1.)		 

@click.command()
@click.argument("env_id", type=str, default="chakra")
def main(env_id):
    # Register the environment
    rng = np.random.RandomState(42)

    if env_id == 'chakra':
        from rlpa2 import chakra
        env = gym.make('chakra')
        get_action = chakra_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError(
            "Unsupported environment: must be 'chakra' ")

    env.seed(42)

    # Initialize parameters
    theta = rng.normal(scale=0.01, size=(action_dim, obs_dim + 1))

    while True:
        ob = env.reset()
        done = False
        # Only render the first trajectory
        # Collect a new trajectory
        rewards = []
        while not done:
            action = get_action(theta, ob, rng=rng)
            next_ob, rew, done, _ = env.step(action)
            ob = next_ob
            env.render()
            rewards.append(rew)

        print("Episode reward: %.2f" % np.sum(rewards))

if __name__ == "__main__":
    main()
