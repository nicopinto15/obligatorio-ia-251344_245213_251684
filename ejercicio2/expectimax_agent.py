import numpy as np
from agent import Agent
import time

class ExpectimaxAgent(Agent):
    def __init__(self, env, depth=4):
        self.env = env
        self.depth = depth
        self.nodes_explored = 0
        
    def get_valid_actions(self, board):
        """Get all valid actions for the current board state"""
        actions = []
        size = board.shape[0]
        for is_row in [0, 1]:
            for idx in range(size):
                line = board[idx, :] if is_row else board[:, idx]
                start = None
                for i in range(size):
                    if line[i] == 1:
                        if start is None:
                            start = i
                    elif start is not None:
                        actions.append([idx, start, i-1, is_row])
                        start = None
                if start is not None:
                    actions.append([idx, start, size-1, is_row])
        return actions
    
    def apply_action(self, board, action):
        """Apply an action to the board and return new board state"""
        new_board = board.copy()
        idx, start, end, is_row = action
        if is_row:
            new_board[idx, start:end+1] = 0
        else:
            new_board[start:end+1, idx] = 0
        return new_board
    
    def is_terminal(self, board):
        """Check if the game is over (no pieces left)"""
        return np.count_nonzero(board) == 0
    
    def evaluate_board(self, board, maximizing_player):
        """Evaluate the board state using multiple heuristics"""
        if self.is_terminal(board):
            return -1000 if maximizing_player else 1000
        # Combine multiple evaluation functions
        nim_value = self.nim_sum_evaluation(board)
        piece_count = self.piece_count_evaluation(board)
        mobility = self.mobility_evaluation(board)
        structure = self.structure_evaluation(board)
        # Weighted combination of heuristics (normal play)
        evaluation = 0.4 * nim_value + 0.2 * piece_count + 0.3 * mobility + 0.1 * structure
        return evaluation
    
    def nim_sum_evaluation(self, board):
        """Evaluate based on Nim-sum (XOR of row/column sizes)"""
        nim_sum = 0
        size = board.shape[0]
        
        for is_row in [0, 1]:
            for idx in range(size):
                line = board[idx, :] if is_row else board[:, idx]
                current_length = 0
                for i in range(size):
                    if line[i] == 1:
                        current_length += 1
                    elif current_length > 0:
                        nim_sum ^= current_length
                        current_length = 0
                if current_length > 0:
                    nim_sum ^= current_length
        
        return -100 if nim_sum == 0 else nim_sum
    
    def piece_count_evaluation(self, board):
        """Simple piece count evaluation"""
        return np.count_nonzero(board)
    
    def mobility_evaluation(self, board):
        """Evaluate based on number of available moves"""
        return len(self.get_valid_actions(board))
    
    def structure_evaluation(self, board):
        """Evaluate board structure"""
        size = board.shape[0]
        structure_score = 0
        
        for i in range(size):
            for j in range(size):
                if board[i, j] == 1:
                    neighbors = 0
                    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size and board[ni, nj] == 1:
                            neighbors += 1
                    if neighbors == 0:
                        structure_score -= 5
                    else:
                        structure_score += neighbors
        
        return structure_score
    
    def get_action_probabilities(self, board):
        """Get probability distribution over actions based on heuristics"""
        actions = self.get_valid_actions(board)
        if not actions:
            return []
        
        # Calculate scores for each action
        scores = []
        for action in actions:
            new_board = self.apply_action(board, action)
            score = self.evaluate_board(new_board, False)  # Opponent's perspective
            scores.append(score)
        
        # Convert to probabilities using softmax
        scores = np.array(scores)
        # Invert scores since we want higher probability for worse moves (from opponent's perspective)
        scores = -scores
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        probabilities = exp_scores / np.sum(exp_scores)
        
        return list(zip(actions, probabilities))
    
    def expectimax(self, board, depth, maximizing_player):
        """Expectimax algorithm"""
        self.nodes_explored += 1
        
        if depth == 0 or self.is_terminal(board):
            return self.evaluate_board(board, maximizing_player), None
        
        valid_actions = self.get_valid_actions(board)
        if not valid_actions:
            return self.evaluate_board(board, maximizing_player), None
        
        if maximizing_player:
            # Maximizing player: choose best action
            max_eval = float('-inf')
            best_action = None
            
            for action in valid_actions:
                new_board = self.apply_action(board, action)
                eval_score, _ = self.expectimax(new_board, depth - 1, False)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
            
            return max_eval, best_action
        else:
            # Chance node: compute expected value
            action_probs = self.get_action_probabilities(board)
            expected_value = 0
            
            for action, prob in action_probs:
                new_board = self.apply_action(board, action)
                eval_score, _ = self.expectimax(new_board, depth - 1, True)
                expected_value += prob * eval_score
            
            return expected_value, None
    
    def act(self, obs):
        """Choose the best action using expectimax"""
        self.nodes_explored = 0
        start_time = time.time()
        
        board = obs["board"]
        _, best_action = self.expectimax(board, self.depth, True)
        
        end_time = time.time()
        
        # Debug information
        print(f"Expectimax explored {self.nodes_explored} nodes in {end_time - start_time:.3f} seconds")
        
        return best_action if best_action else self.get_valid_actions(board)[0]
