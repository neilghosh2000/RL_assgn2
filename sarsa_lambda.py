import gym
import GridWorld
import random
import matplotlib.pyplot as plt

n_iters = 2500
n_episodes = 50
goal = [0, 11]


lambdas = [0, 0.3, 0.5, 0.9, 0.99, 1.0]

fig_rewards = plt.figure().add_subplot(111)
fig_steps = plt.figure().add_subplot(111)

for k in range(len(lambdas)):
    curr_lambda = lambdas[k]
    reward_list = list()
    step_list = list()
    env = gym.make('PuddleWorld-v0', goal=goal, algorithm='sarsa_l', lambda_l=curr_lambda)
    for i in range(n_iters):
        reward_list.append(list())
        step_list.append(list())
    for i in range(n_episodes):
        env.reset()
        for j in range(n_iters):
            curr_steps, reward = env.step('action')
            if j == 1:
                print(reward, curr_steps)
            reward_list[j].append(reward)
            step_list[j].append(curr_steps)
        env.step(action='update')

    avg_rewards = list()
    avg_steps = list()

    env.render()

    for i in range(n_iters):
        avg_rewards.append(sum(reward_list[i]) / n_episodes)
        avg_steps.append(sum(step_list[i]) / n_episodes)

    fig_rewards.plot(range(n_iters), avg_rewards, label=r'$\lambda$ = ' + str(curr_lambda))
    fig_steps.plot(range(n_iters), avg_steps, label=r'$\lambda$ = ' + str(curr_lambda))


fig_rewards.set_xlabel('Iterations')
fig_rewards.set_ylabel('Average Reward')
fig_rewards.title.set_text('Average Reward vs Time')
fig_rewards.legend(loc='lower right')

fig_steps.set_xlabel('Iterations')
fig_steps.set_ylabel('Average Episode Length')
fig_steps.title.set_text('Average Episode Length vs Time')
fig_steps.legend(loc='upper right')

plt.show()