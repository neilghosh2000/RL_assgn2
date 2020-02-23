def sarsa(self):
    start_index = random.randint(0, 3)
    current_state = [self.start_coords[start_index], 0]
    curr_row = current_state[0]
    curr_col = current_state[1]
    total_reward = 0
    step = 1
    curr_action = 0
    start_q = self.q_values[(curr_row, curr_col)]
    while True:
        x = random.uniform(0, 1)
        if x > self.epsilon:
            curr_action = start_q.index(max(start_q))
        else:
            curr_action = random.randint(0, 3)
        if self.is_valid_action(curr_row, curr_col, curr_action):
            break

    curr_q = start_q[curr_action]
    next_row, next_col = self.get_next_state(curr_row, curr_col, curr_action)
    reward = self.world[curr_row][curr_col]
    total_reward += reward
    curr_q, next_action = self.sarsa_q_update(next_row, next_col, curr_q, reward)
    start_q[curr_action] = curr_q
    self.q_values[(curr_row, curr_col)] = start_q
    curr_row = next_row
    curr_col = next_col
    curr_action = next_action

    while True:
        if curr_row == self.goal[0] and curr_col == self.goal[1]:
            break
        step += 1
        x = random.uniform(0, 1)
        q_list = self.q_values[(curr_row, curr_col)]
        if x > self.transition_prob:
            curr_q = q_list[curr_action]
            next_row, next_col = self.get_next_state(curr_row, curr_col, curr_action)
            reward = self.world[next_row][next_col]
            total_reward += reward
            curr_q, next_action = self.sarsa_q_update(next_row, next_col, curr_q, reward)
            q_list[curr_action] = curr_q
            self.q_values[(curr_row, curr_col)] = q_list
            curr_row = next_row
            curr_col = next_col
            curr_action = next_action
        else:
            random_action = self.get_random_action(curr_action)
            if self.is_valid_action(curr_row, curr_col, random_action):
                curr_q = q_list[curr_action]
                next_row, next_col = self.get_next_state(curr_row, curr_col, random_action)
                reward = self.world[next_row][next_col]
                total_reward += reward
                curr_q, next_action = self.sarsa_q_update(next_row, next_col, curr_q, reward)
                q_list[curr_action] = curr_q
                self.q_values[(curr_row, curr_col)] = q_list
                curr_row = next_row
                curr_col = next_col
                curr_action = next_action

    return step, total_reward