import random
import pickle
import numpy as np

class TicTacToeAI:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.player = 'O'
        self.opponent = 'X'

    def get_state_key(self, board):
        return ''.join(board)

    def available_moves(self, board):
        return [i for i, cell in enumerate(board) if cell == ' ']

    def choose_action(self, board):
        state = self.get_state_key(board)
        if random.uniform(0, 1) < self.epsilon or state not in self.q_table:
            return random.choice(self.available_moves(board))
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def learn(self, old_state, action, reward, new_state, done):
        old_key = self.get_state_key(old_state)
        new_key = self.get_state_key(new_state)
        if old_key not in self.q_table:
            self.q_table[old_key] = {a: 0 for a in range(9)}
        if new_key not in self.q_table:
            self.q_table[new_key] = {a: 0 for a in range(9)}

        predict = self.q_table[old_key][action]
        target = reward if done else reward + self.gamma * max(self.q_table[new_key].values())
        self.q_table[old_key][action] += self.alpha * (target - predict)

    def check_winner(self, board):
        wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        for cond in wins:
            if board[cond[0]] != ' ' and all(board[cond[0]] == board[i] for i in cond):
                return board[cond[0]]
        if ' ' not in board:
            return 'Draw'
        return None

    def train(self, episodes=50000):
        for _ in range(episodes):
            board = [' '] * 9
            done = False
            current_player = self.player

            while not done:
                state = board.copy()
                moves = self.available_moves(board)
                if current_player == self.player:
                    action = self.choose_action(board)
                else:
                    action = random.choice(moves)

                board[action] = current_player
                winner = self.check_winner(board)

                if winner == self.player:
                    reward = 1; done = True
                elif winner == self.opponent:
                    reward = -1; done = True
                elif winner == 'Draw':
                    reward = 0.5; done = True
                else:
                    reward = 0

                new_state = board.copy()
                self.learn(state, action, reward, new_state, done)
                current_player = self.opponent if current_player == self.player else self.player

        with open('q_table.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)

    def best_move(self, board):
        state = self.get_state_key(board)
        if state in self.q_table:
            valid_moves = self.available_moves(board)
            q_vals = {a: self.q_table[state].get(a, -999) for a in valid_moves}
            return max(q_vals, key=q_vals.get)
        return random.choice(self.available_moves(board))

if __name__ == "__main__":
    ai = TicTacToeAI()
    ai.train()
