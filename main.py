import gym
import GridWorld
import random
import matplotlib.pyplot as plt

goal = [0, 11]
env = gym.make('PuddleWorld-v0', goal=goal, algorithm='sarsa')
reward_list = list()
step_list = list()
for i in range(2000):
    steps = 0
    rewards = 0
    for j in range(50):
        print(i, j)
        curr_steps, reward = env.step('action')
        steps += curr_steps
        rewards += reward
    reward_list.append(rewards / 50)
    step_list.append(steps / 50)

env.render(mode='human')
print(step_list[-1])
fig_rewards = plt.figure().add_subplot(111)
fig_steps = plt.figure().add_subplot(111)

fig_rewards.set_xlabel('Iterations')
fig_rewards.set_ylabel('Average Reward')
fig_rewards.title.set_text('Average Reward vs Time')

fig_steps.set_xlabel('Iterations')
fig_steps.set_ylabel('Average Episode Length')
fig_steps.title.set_text('Average Episode Length vs Time')

fig_rewards.plot(range(2000), reward_list)
fig_steps.plot(range(2000), step_list)

plt.show()
