import pandas as pd
import numpy as np

class WarehouseEnv:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.q_vals = np.zeros((rows, cols, 4))
        self.actions = ['up', 'down', 'left', 'right']
        self.rewards = np.full((rows, cols), -100)
        self.rewards = self.define_aisles()
        self.rewards[0, 5] = 100
        self.terminal_state = False

    def define_aisles(self):
        aisles = {}  # store locations in a dictionary
        aisles[1] = [i for i in range(1, 10)]
        aisles[2] = [1, 7, 9]
        aisles[3] = [i for i in range(1, 8)]
        aisles[3].append(9)
        aisles[4] = [3, 7]
        aisles[5] = [i for i in range(11)]
        aisles[6] = [5]
        aisles[7] = [i for i in range(1, 10)]
        aisles[8] = [3, 7]
        aisles[9] = [i for i in range(11)]

        for row_index in range(1, 10):
            for column_index in aisles[row_index]:
                self.rewards[row_index, column_index] = -1
        return self.rewards

    def is_terminal_state(self, current_row_index, current_column_index):
        # if the reward for this location is -1, then it is not a terminal state (i.e., it is a 'white square')
        if self.rewards[current_row_index, current_column_index] == -1.:
            return False
        else:
            return True

    def get_next_action(self, current_row_index, current_column_index, epsilon):
        # if a randomly chosen value between 0 and 1 is less than epsilon,
        # then choose the most promising value from the Q-table for this state.
        if np.random.random() < epsilon:
            return np.argmax(self.q_vals[current_row_index, current_column_index])
        else:  # choose a random action
            return np.random.randint(4)

    # define a function that will choose a random, non-terminal starting location
    def get_starting_location(self):
        # get a random row and column index
        current_row_index = np.random.randint(self.rows)
        current_column_index = np.random.randint(self.cols)
        # continue choosing random row and column indexes until a non-terminal state is identified
        # (i.e., until the chosen state is a 'white square').
        while self.is_terminal_state(current_row_index, current_column_index):
            current_row_index = np.random.randint(self.rows)
            current_column_index = np.random.randint(self.cols)
        return current_row_index, current_column_index

    def get_next_location(self, current_row_index, current_column_index, action_index):
        new_row_index = current_row_index
        new_column_index = current_column_index
        if self.actions[action_index] == 'up' and current_row_index > 0:
            new_row_index -= 1
        elif self.actions[action_index] == 'right' and current_column_index < self.cols - 1:
            new_column_index += 1
        elif self.actions[action_index] == 'down' and current_row_index < self.rows - 1:
            new_row_index += 1
        elif self.actions[action_index] == 'left' and current_column_index > 0:
            new_column_index -= 1
        return new_row_index, new_column_index

    def get_shortest_path(self,start_row_index, start_column_index):
        # return immediately if this is an invalid starting location
        if self.is_terminal_state(start_row_index, start_column_index):
            return []
        else:  # if this is a 'legal' starting location
            current_row_index, current_column_index = start_row_index, start_column_index
            shortest_path = []
            shortest_path.append([current_row_index, current_column_index])
            # continue moving along the path until we reach the goal (i.e., the item packaging location)
            while not self.is_terminal_state(current_row_index, current_column_index):
                # get the best action to take
                action_index = self.get_next_action(current_row_index, current_column_index, 1.)
                # move to the next location on the path, and add the new location to the list
                current_row_index, current_column_index = self.get_next_location(current_row_index, current_column_index,
                                                                            action_index)
                shortest_path.append([current_row_index, current_column_index])
            return shortest_path

    def train_agent(self, epsilon, discount_factor, learning_rate, trials):
        for episode in range(trials):
            # get the starting location for this episode
            row_index, column_index = self.get_starting_location()

            # continue taking actions (i.e., moving) until we reach a terminal state
            # (i.e., until we reach the item packaging area or crash into an item storage location)
            while not self.is_terminal_state(row_index, column_index):
                # choose which action to take (i.e., where to move next)
                action_index = self.get_next_action(row_index, column_index, epsilon)

                # perform the chosen action, and transition to the next state (i.e., move to the next location)
                old_row_index, old_column_index = row_index, column_index  # store the old row and column indexes
                row_index, column_index = self.get_next_location(row_index, column_index, action_index)

                # receive the reward for moving to the new state, and calculate the temporal difference
                reward = self.rewards[row_index, column_index]
                old_q_value = self.q_vals[old_row_index, old_column_index, action_index]
                temporal_difference = reward + (
                            discount_factor * np.max(self.q_vals[row_index, column_index])) - old_q_value

                # update the Q-value for the previous state and action pair
                new_q_value = old_q_value + (learning_rate * temporal_difference)
                self.q_vals[old_row_index, old_column_index, action_index] = new_q_value

        print('Training complete!')
agent = WarehouseEnv(11, 11)
agent.train_agent(.9, .9, .9, 1000)

print(agent.get_shortest_path(3, 9)) #starting at row 3, column 9
print(agent.get_shortest_path(5, 0)) #starting at row 5, column 0
print(agent.get_shortest_path(9, 5)) #starting at row 9, column 5

#display an example of reversed shortest path
path = agent.get_shortest_path(5, 2) #go to row 5, column 2
path.reverse()
print(path)