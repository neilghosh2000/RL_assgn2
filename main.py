import gym
import GridWorld
import random
import matplotlib.pyplot as plt

n_iters = 3000
n_episodes = 50
goal = [2, 9]
reward_list = list()
step_list = list()

for i in range(n_iters):
    reward_list.append(list())
    step_list.append(list())

env = gym.make('PuddleWorld-v0', goal=goal, algorithm='q', lambda_l=0, wind=True)

for i in range(n_episodes):
    env.reset()
    print(i)
    for j in range(n_iters):
        curr_steps, reward = env.step('action')
        reward_list[j].append(reward)
        step_list[j].append(curr_steps)
    env.step(action='update')

avg_rewards = list()
avg_steps = list()
env.render()

for i in range(n_iters):
    avg_rewards.append(sum(reward_list[i]) / n_episodes)
    avg_steps.append(sum(step_list[i]) / n_episodes)

fig_rewards = plt.figure().add_subplot(111)
fig_steps = plt.figure().add_subplot(111)

fig_rewards.set_xlabel('Iterations')
fig_rewards.set_ylabel('Average Reward')
fig_rewards.title.set_text('Average Reward vs Time')

fig_steps.set_xlabel('Iterations')
fig_steps.set_ylabel('Average Episode Length')
fig_steps.title.set_text('Average Episode Length vs Time')

fig_rewards.plot(range(n_iters), avg_rewards)
fig_steps.plot(range(n_iters), avg_steps)

plt.show()
