import gym
import random
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class GridWorld(gym.Env):
    def __init__(self, goal, algorithm, lambda_l, wind):
        print('Initialized')
        self.algo = algorithm
        self.wind = wind
        self.world = list()
        self.q_values = {}
        self.rows = 12
        self.cols = 12
        self.actions = ['up', 'down', 'left', 'right']
        self.q_total = {}
        self.eligibility_trace = {}
        for i in range(12):
            temp = list()
            for j in range(12):
                q_list = list()
                q_t = list()
                e = list()
                temp.append(-0.05)
                for k in range(0, 4):
                    q_list.append(0)
                    q_t.append(0)
                    e.append(0)
                self.q_values[(i, j)] = q_list
                self.q_total[(i, j)] = q_t
                self.eligibility_trace[(i, j)] = e
            self.world.append(temp)
        # for i in range(12):
        #     self.q_values[(0, i)][0] = -1
        #     self.q_values[(11, i)][1] = -1
        #     self.q_values[(i, 0)][2] = -1
        #     self.q_values[(i, 11)][3] = -1
        self.goal = goal
        self.start_coords = [5, 6, 10, 11]
        self.epsilon = 0.1
        self.gamma = 0.9
        self.transition_prob = 0.1
        self.alpha = 0.1
        self.eligibility_lambda = lambda_l
        self.setup_world()

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
        if x > 0.5 and self.wind:
            col_shift = 1
        if action == 0:
            # if col_shift == 1 and col + 1 < self.cols:
            #     return row-1, col+1
            # else:
            #     return row-1, col
            return row - 1, col + col_shift
        elif action == 1:
            # if col_shift == 1 and col + 1 < self.cols:
            #     return row+1, col + 1
            # else:
            #     return row+1, col
            return row + 1, col + col_shift
        elif action == 2:
            # if col_shift == 1 and col < self.cols:
            #     return row, col
            # else:
            #     return row, col-1
            return row, col - 1 + col_shift
        else:
            # if col_shift == 1 and col + 2 < self.cols:
            #     return row, col+2
            # else:
            #     return row, col+1
            return row, col + 1 + col_shift

    def get_random_action(self, action):
        new_action_list = list()
        for i in range(4):
            if i != action:
                new_action_list.append(i)
        x = random.randint(0, 2)
        return new_action_list[x]

    def is_invalid_state(self, row, col):
        return row < 0 or row >= self.rows or col < 0 or col >= self.cols

    def step(self, action):
        if action == 'update':
            return self.add_q_values(self.q_values)
        else:
            if self.algo == 'sarsa':
                return self.sarsa()
            elif self.algo == 'sarsa_l':
                return self.sarsa_l()
            else:
                return self.q_learning()

    def q_learning(self):
        start_index = random.randint(0, 3)
        current_state = [self.start_coords[start_index], 0]
        total_reward = 0
        step = 0
        while True:
            curr_row = current_state[0]
            curr_col = current_state[1]
            step += 1
            # print('Row: ' + str(curr_row) + ',' + 'Col: ' + str(curr_col))
            if curr_row == self.goal[0] and curr_col == self.goal[1]:
                break
            x = random.uniform(0, 1)
            q_list = self.q_values[(curr_row, curr_col)]
            if x > self.epsilon:
                action = q_list.index(max(q_list))
            else:
                action = random.randint(0, 3)

            x = random.uniform(0, 1)
            if x > self.transition_prob:
                next_row, next_col = self.get_next_state(curr_row, curr_col, action)
                if not self.is_invalid_state(next_row, next_col):
                    curr_q_val = q_list[action]
                    reward = self.world[next_row][next_col]
                    total_reward += reward
                    curr_q_val = self.update_q_val(next_row, next_col, curr_q_val, reward)
                    q_list[action] = curr_q_val
                    current_state = [next_row, next_col]
                else:
                    current_state = [curr_row, curr_col]
            else:
                random_action = self.get_random_action(action)
                next_row, next_col = self.get_next_state(curr_row, curr_col, random_action)
                if not self.is_invalid_state(next_row, next_col):
                    curr_q_val = q_list[action]
                    reward = self.world[next_row][next_col]
                    total_reward += reward
                    curr_q_val = self.update_q_val(next_row, next_col, curr_q_val, reward)
                    q_list[action] = curr_q_val
                    current_state = [next_row, next_col]
                else:
                    current_state = [curr_row, curr_col]
                # if self.is_valid_action(curr_row, curr_col, random_action):
                #     next_row, next_col = self.get_next_state(curr_row, curr_col, random_action)
                #     curr_q_val = q_list[action]
                #     reward = self.world[next_row][next_col]
                #     total_reward += reward
                #     curr_q_val = self.sarsa_q_update(next_row, next_col, curr_q_val, reward)
                #     q_list[action] = curr_q_val
                #     self.q_values[(curr_row, curr_col)] = q_list
                #     current_state = [next_row, next_col]

        return step, total_reward

    def update_q_val(self, next_row, next_col, curr_q, reward):
        next_q = self.q_values[(next_row, next_col)]
        q_max = max(next_q)
        curr_q = curr_q + self.alpha * (reward + self.gamma * q_max - curr_q)
        return curr_q

    def sarsa(self):
        start_index = random.randint(0, 3)
        current_state = [self.start_coords[start_index], 0]
        total_reward = 0
        step = 0
        while True:
            curr_row = current_state[0]
            curr_col = current_state[1]
            step += 1
            # print('Row: ' + str(curr_row) + ',' + 'Col: ' + str(curr_col))
            if curr_row == self.goal[0] and curr_col == self.goal[1]:
                break
            x = random.uniform(0, 1)
            q_list = self.q_values[(curr_row, curr_col)]
            if x > self.epsilon:
                action = q_list.index(max(q_list))
            else:
                action = random.randint(0, 3)

            x = random.uniform(0, 1)
            if x > self.transition_prob:
                next_row, next_col = self.get_next_state(curr_row, curr_col, action)
                if not self.is_invalid_state(next_row, next_col):
                    curr_q_val = q_list[action]
                    reward = self.world[next_row][next_col]
                    total_reward += reward
                    curr_q_val = self.sarsa_q_update(next_row, next_col, curr_q_val, reward)
                    q_list[action] = curr_q_val
                    current_state = [next_row, next_col]
                else:
                    current_state = [curr_row, curr_col]
            else:
                random_action = self.get_random_action(action)
                next_row, next_col = self.get_next_state(curr_row, curr_col, random_action)
                if not self.is_invalid_state(next_row, next_col):
                    curr_q_val = q_list[action]
                    reward = self.world[next_row][next_col]
                    total_reward += reward
                    curr_q_val = self.sarsa_q_update(next_row, next_col, curr_q_val, reward)
                    q_list[action] = curr_q_val
                    current_state = [next_row, next_col]
                else:
                    current_state = [curr_row, curr_col]
                # if self.is_valid_action(curr_row, curr_col, random_action):
                #     next_row, next_col = self.get_next_state(curr_row, curr_col, random_action)
                #     curr_q_val = q_list[action]
                #     reward = self.world[next_row][next_col]
                #     total_reward += reward
                #     curr_q_val = self.sarsa_q_update(next_row, next_col, curr_q_val, reward)
                #     q_list[action] = curr_q_val
                #     self.q_values[(curr_row, curr_col)] = q_list
                #     current_state = [next_row, next_col]

        return step, total_reward

    def sarsa_q_update(self, next_row, next_col, curr_q, reward):
        next_q = self.q_values[(next_row, next_col)]
        # while True:
        #     x = random.uniform(0, 1)
        #     if x > self.epsilon:
        #         next_action = next_q.index(max(next_q))
        #     else:
        #         next_action = random.randint(0, 3)
        #     if self.is_valid_action(next_row, next_col, next_action):
        #         break
        x = random.uniform(0, 1)
        if x > self.epsilon:
            next_action = next_q.index(max(next_q))
        else:
            next_action = random.randint(0, 3)
        curr_q = curr_q + self.alpha * (reward + self.gamma * next_q[next_action] - curr_q)
        return curr_q

    def sarsa_l(self):
        start_index = random.randint(0, 3)
        current_state = [self.start_coords[start_index], 0]
        total_reward = 0
        step = 0
        while True:
            step += 1
            curr_row = current_state[0]
            curr_col = current_state[1]
            # print('Row: ' + str(curr_row) + ',' + 'Col: ' + str(curr_col))
            if curr_row == self.goal[0] and curr_col == self.goal[1]:
                break
            x = random.uniform(0, 1)
            q_list = self.q_values[(curr_row, curr_col)]
            if x > self.epsilon:
                action = q_list.index(max(q_list))
            else:
                action = random.randint(0, 3)

            x = random.uniform(0, 1)
            if x > self.transition_prob:
                next_row, next_col = self.get_next_state(curr_row, curr_col, action)
                if not self.is_invalid_state(next_row, next_col):
                    curr_q_val = q_list[action]
                    reward = self.world[next_row][next_col]
                    total_reward += reward
                    delta = self.get_delta(next_row, next_col, curr_q_val, reward)
                    self.sarsa_lambda_update(delta)
                    current_state = [next_row, next_col]
                else:
                    current_state = [curr_row, curr_col]
            else:
                random_action = self.get_random_action(action)
                next_row, next_col = self.get_next_state(curr_row, curr_col, random_action)
                if not self.is_invalid_state(next_row, next_col):
                    curr_q_val = q_list[action]
                    reward = self.world[next_row][next_col]
                    total_reward += reward
                    delta = self.get_delta(next_row, next_col, curr_q_val, reward)
                    self.sarsa_lambda_update(delta)
                    current_state = [next_row, next_col]
                else:
                    current_state = [curr_row, curr_col]

        return step, total_reward

    def get_delta(self, next_row, next_col, curr_q, reward):
        next_q = self.q_values[(next_row, next_col)]
        while True:
            x = random.uniform(0, 1)
            if x > self.epsilon:
                next_action = next_q.index(max(next_q))
            else:
                next_action = random.randint(0, 3)
            if self.is_valid_action(next_row, next_col, next_action):
                break
        delta = reward + self.gamma*next_q[next_action] - curr_q
        return delta

    def sarsa_lambda_update(self, delta):
        for i in range(self.rows):
            for j in range(self.cols):
                q_list = self.q_values[(i, j)]
                trace_list = self.eligibility_trace[(i, j)]
                for k in range(0, 4):
                    q_list[k] = q_list[k] + self.alpha*trace_list[k]*delta

    def update_trace(self, row, col, action):
        for i in range(self.rows):
            for j in range(self.cols):
                trace_list = self.eligibility_trace[(i, j)]
                for k in range(0, 4):
                    if i == row and j == col and k == action:
                        trace_list[k] = self.gamma*self.eligibility_lambda*trace_list[k] + 1
                    elif i == row and j == col and k != action:
                        trace_list[k] = 0
                    else:
                        trace_list[k] = self.gamma*self.eligibility_lambda*trace_list[k]
                self.eligibility_trace[(i, j)] = trace_list

    def add_q_values(self, new_q_list):
        for i in range(12):
            for j in range(12):
                curr_q_t = self.q_total[(i, j)]
                new_q_l = new_q_list[(i, j)]
                for k in range(4):
                    curr_q_t[k] += new_q_l[k]
        return True

    def reset(self):
        for i in range(12):
            for j in range(12):
                q_list = list()
                for k in range(0, 4):
                    q_list.append(0)
                self.q_values[(i, j)] = q_list
        # for i in range(12):
        #     self.q_values[(0, i)][0] = -10
        #     self.q_values[(11, i)][1] = -10
        #     self.q_values[(i, 0)][2] = -10
        #     self.q_values[(i, 11)][3] = -10

    def render(self, mode='human'):
        print(self.q_total)
        self.get_optimal_policy()

    def get_optimal_policy(self):
        plt.figure(figsize=(12, 12))
        color_map = colors.ListedColormap(
            [(0, 0, 0), (0.3, 0.3, 0.3), (0.6, 0.6, 0.6), (0.9, 0.9, 0.9), (1, 0, 0), (0, 0, 0.8)])
        plt.pcolor(self.world, edgecolors='k', linewidth=5)
        ax = plt.gca()
        ax.invert_yaxis()
        matrix_s = ''
        for i in range(12):
            row_s = '|'
            for j in range(12):
                s = ""
                q_list = self.q_total[(j, i)]
                opt_action = q_list.index(max(q_list))
                if opt_action == 0:
                    s += '^'
                    ax.arrow(i + 0.5, j + 0.8, 0, -0.4, head_width=0.25, head_length=0.25, fc='g', ec='g')
                elif opt_action == 1:
                    s += 'v'
                    ax.arrow(i + 0.5, j + 0.2, 0, 0.4, head_width=0.25, head_length=0.25, fc='g', ec='g')
                elif opt_action == 2:
                    s += '<'
                    ax.arrow(i + 0.8, j + 0.5, -0.4, 0, head_width=0.25, head_length=0.25, fc='g', ec='g')
                else:
                    s += '>'
                    ax.arrow(i + 0.2, j + 0.5, 0.4, 0, head_width=0.25, head_length=0.25, fc='g', ec='g')
                s += '|'
                row_s += s
            row_s += '\n'
            matrix_s += row_s

        print(matrix_s)
        plt.show()
