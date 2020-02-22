import gym
import random


class GridWorld(gym.Env):
    def __init__(self, goal):
        print('Initialized')
        self.world = list()
        self.q_values = {}
        self.rows = 12
        self.cols = 12
        self.actions = ['up', 'down', 'left', 'right']
        for i in range(12):
            temp = list()
            q_list = list()
            for j in range(12):
                temp.append(0)
                for k in range(0, 4):
                    q_list.append(0)
                self.q_values[(i, j)] = q_list
            self.world.append(temp)
        self.goal = goal
        self.start_coords = [5, 6, 10, 11]
        self.epsilon = 0.1
        self.gamma = 0.9
        self.transition_prob = 0.9
        self.alpha = 0.1
        self.setup_world()
        for i in range(12):
            print(self.world[i])

    def step(self, action):
        start_index = random.randint(0, 3)
        start_state = [self.start_coords[start_index], 0]
        current_state = [self.start_coords[start_index], 0]
        total_reward = 0
        step = 0
        while True:
            step += 1
            curr_row = current_state[0]
            curr_col = current_state[1]
            if curr_row == self.goal[0] and curr_col == self.goal[1]:
                print('Reached')
                break
            x = random.uniform(0, 1)
            q_list = self.q_values[(curr_row, curr_col)]
            if x > self.epsilon:
                action = q_list.index(max(q_list))
            else:
                action = random.randint(0, 3)

            if self.is_valid_action(curr_row, curr_col, action):
                x = random.uniform(0, 1)
                if x > self.transition_prob:
                    next_row, next_col = self.get_next_state(curr_row, curr_col, action)
                    curr_q_val = q_list[action]
                    reward = self.world[next_row][next_col]
                    total_reward += reward
                    curr_q_val = self.update_q_val(next_row, next_col, curr_q_val, reward)
                    q_list[action] = curr_q_val
                    self.q_values[(curr_row, curr_col)] = q_list
                    current_state = [next_row, next_col]
                else:
                    random_action = self.get_random_action(action)
                    if self.is_valid_action(curr_row, curr_col, random_action):
                        next_row, next_col = self.get_next_state(curr_row, curr_col, action)
                        curr_q_val = q_list[action]
                        reward = self.world[next_row][next_col]
                        total_reward += reward
                        curr_q_val = self.update_q_val(next_row, next_col, curr_q_val, reward)
                        q_list[action] = curr_q_val
                        self.q_values[(curr_row, curr_col)] = q_list
                        current_state = [next_row, next_col]

        return step

    def is_valid_action(self, row, col, action):
        if action == 0:
            if row-1 >= 0:
                return True
            else:
                return False
        if action == 1:
            if row+1 < self.rows:
                return True
            else:
                return False
        if action == 2:
            if col-1 >= 0:
                return True
            else:
                return False
        else:
            if col+1 < self.cols:
                return True
            else:
                return False

    def get_next_state(self, row, col, action):
        x = random.uniform(0, 1)
        col_shift = 0
        if x > 0.5:
            col_shift = 1

        if action == 0:
            if col_shift == 1 and col + 1 < self.cols:
                return row-1, col+1
            else:
                return row-1, col
        if action == 1:
            if col_shift == 1 and col + 1 < self.cols:
                return row +1, col + 1
            else:
                return row +1, col
        if action == 2:
            if col_shift == 1 and col + 1 < self.cols:
                return row, col
            else:
                return row, col-1
        else:
            if col_shift == 1 and col + 2 < self.cols:
                return row, col+2
            else:
                return row, col+1

    def update_q_val(self, next_row, next_col, curr_q, reward):
        next_q = self.q_values[(next_row, next_col)]
        q_max = max(next_q)
        curr_q = curr_q + self.alpha*(reward + self.gamma*q_max - curr_q)
        return curr_q

    def get_random_action(self, action):
        new_action_list = list()
        for i in range(4):
            if i != action:
                new_action_list.append(i)
        x = random.randint(0, 2)
        return new_action_list[x]

    def reset(self):
        print('Reset successful')

    def render(self, mode='human'):
        print('Render successful')

    def setup_world(self):
        for i in range(0, 3):
            self.add_rewards(2+i, 8-i, 3+i, True, -(i+1))
            self.add_rewards(3+i, 8-i, 2+i, False, -(i+1))
            self.add_rewards(2+i, 6-i, 8-i, True, -(i+1))
            self.add_rewards(7-i, 8-i, 6-i, False, -(i+1))
            self.add_rewards(6-i, 8-i, 7-i, True, -(i+1))
            self.add_rewards(3+i, 7-i, 8-i, False, -(i+1))

        self.world[self.goal[0]][self.goal[1]] = 10

    def add_rewards(self, start, end, fixed, vertical, reward):
        for i in range(start, end+1):
            if vertical:
                self.world[i][fixed] = reward
            else:
                self.world[fixed][i] = reward
