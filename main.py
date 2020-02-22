import gym
import GridWorld
import random
import matplotlib.pyplot as plt

goal = [0, 11]
env = gym.make('PuddleWorld-v0', goal=goal)
reward_list = list()
step_list = list()
for i in range(50):
    steps = env.step('action')
    # reward_list.append(reward)
    step_list.append(steps)
env.reset()

plt.plot(range(50), step_list)
plt.show()
