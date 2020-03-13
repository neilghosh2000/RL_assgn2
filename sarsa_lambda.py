import gym
import GridWorld
import random
import matplotlib.pyplot as plt

n_iters = 25
n_episodes = 50
goal = [2, 9]


lambdas = [0, 0.3, 0.5, 0.9, 0.99, 1.0]

fig_rewards = plt.figure().add_subplot(111)
fig_steps = plt.figure().add_subplot(111)

avg_rewards = list()
avg_steps = list()

for i in range(len(lambdas)):
    curr_lambda = lambdas[i]
    curr_rewards = list()
    curr_steps = list()
    env = gym.make('PuddleWorld-v0', goal=goal, algorithm='sarsa_l', lambda_l=curr_lambda, wind=True)
    for j in range(n_episodes):
        rewards_list = list()
        steps_list = list()
        print(curr_lambda, j)
        for k in range(n_iters):
            steps, reward = env.step('action')
            rewards_list.append(reward)
            steps_list.append(steps)
        env.step(action='update')
        env.reset()
        curr_rewards.append(rewards_list[-1])
        curr_steps.append(steps_list[-1])
    avg_rewards.append(sum(curr_rewards) / n_episodes)
    avg_steps.append(sum(curr_steps) / n_episodes)

fig_rewards.plot(lambdas, avg_rewards)
fig_steps.plot(lambdas, avg_steps)

fig_rewards.set_xlabel('Lamda')
fig_rewards.set_ylabel('Average Reward')
fig_rewards.title.set_text('Average Reward vs Time')
fig_rewards.legend(loc='lower right')

fig_steps.set_xlabel('Lambda')
fig_steps.set_ylabel('Average Episode Length')
fig_steps.title.set_text('Average Episode Length vs Time')
fig_steps.legend(loc='upper right')

plt.show()
