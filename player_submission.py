#!/usr/bin/env python
from game import Board, game_as_text
from random import randint
import game
import random


class OpenMoveEvalFn:

    def score(self, game, maximizing_player_turn=True):
        return len(game.get_legal_moves()) - len(game.get_opponent_moves())


class CustomEvalFn:
    def __init__(self):
        pass

    def score(self, game, maximizing_player_turn=True):
        my_moves = len(game.get_legal_moves())
        opponent_moves = len(game.get_opponent_moves())
        mobility_score = my_moves - opponent_moves
        center_control_score = self.evaluate_center(game)
        threat_score = self.evaluate_threats(game)

        total_score = mobility_score + center_control_score + threat_score

        return total_score if maximizing_player_turn else -total_score

    def evaluate_center(self, game):
        center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
        score = 0
        for square in center_squares:
            if game.is_spot_open(square[0], square[1]):
                score += 2
            if game.is_spot_queen(square[0], square[1]):
                score += 5
        return score

    def evaluate_threats(self, game):
        """Evaluate if the opponent's queen is threatened."""
        score = 0
        state = game.get_state()
        q = game.get_inactive_players_queen()
        opponent_queen_pos = None
        for row in range(0, 7):
            for col in range(0, 7):
                if state[row][col] == q:
                    opponent_queen_pos = row, col
        if opponent_queen_pos is not None:
            row, col = opponent_queen_pos
            if not game.move_is_in_board(row + 1, col) or not game.move_is_in_board(row - 1, col) or \
               not game.move_is_in_board(row, col + 1) or not game.move_is_in_board(row, col - 1):
                score += 10
        return score


class CustomPlayer:

    def __init__(self, search_depth=7, eval_fn=CustomEvalFn()):
        self.eval_fn = eval_fn
        self.search_depth = search_depth

    def move(self, game, legal_moves, time_left):
        if not legal_moves:
            return None
        self.search_depth = depth(game, self.search_depth)
        depthh = 1
        best_move = random.choice(legal_moves)
        while time_left() > 100 and depthh <= self.search_depth:
            best_move, utility = self.alphabeta(
                game, time_left, depth=self.search_depth)
            depthh += 1
        return best_move

    def utility(self, game, maximizing_player):
        return self.eval_fn.score(game, maximizing_player_turn=maximizing_player)

    def minimax(self, game, time_left, depth, maximizing_player=True):
        # best_move = (0, 0)
        # best_val = float('-inf')

        if not game.get_legal_moves() or depth == 0 or time_left() <= 0:
            return None, self.utility(game, maximizing_player)
        best_move = None
        if maximizing_player:
            best_val = float('-inf')
            for move in game.get_legal_moves():
                next_game, _, _ = game.forecast_move(move)
                # Get value from min_value
                value = self.min_value(
                    next_game, depth - 1, False)
                if value > best_val:
                    best_val = value
                    best_move = move
                    if time_left() <= 0:
                        return best_move, best_val
            return best_move, best_val  # Return value and best move
        else:
            best_val = float('inf')
            for move in game.get_legal_moves():
                next_game, _, _ = game.forecast_move(move)
                # Get value from max_value
                value = self.max_value(
                    next_game,  depth - 1, True)
                if value < best_val:
                    best_val = value
                    best_move = move
                    if time_left() <= 0:
                        return best_move, best_val
            return best_move, best_val  # Return value and best move

    def max_value(self, game, depth, last_best_move):
        # if self.time_left() < TimeLimit
        #   Raise last_best_move

        # Terminal situation: depth = 0 or illegal move
        if depth == 0 or not game.get_legal_moves():
            return self.utility(game, maximizing_player=True)

        # Normal situation: find the maximizing value
        best_score = float('-inf')
        for move in game.get_legal_moves():
            next_game, _, _ = game.forecast_move(move)
            best_score = max(best_score, self.min_value(
                next_game, depth - 1, last_best_move))
        return best_score

    # Minimizing player strategy
    def min_value(self, game, depth, last_best_move):
        # if self.time_left() < TimeLimit
        #   Raise last_best_move

        # Terminal situation: depth = 0 or illegal move
        if depth == 0 or not game.get_legal_moves():
            return self.utility(game, maximizing_player=False)

        # Normal situation: find the minimizing value
        best_score = float('inf')
        for move in game.get_legal_moves():
            next_game, _, _ = game.forecast_move(move)
            best_score = min(best_score, self.max_value(
                next_game, depth - 1, last_best_move))
        return best_score

    def alphabeta(self, game, time_left, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        if depth == 0 or not game.get_legal_moves():
            return None, self.utility(game, maximizing_player)
        best_move = None
        if maximizing_player:
            best_val = float('-inf')
            for move in game.get_legal_moves():
                next_game, _, _ = game.forecast_move(move)

                # Check time left before making a recursive call
                if time_left() <= 0:
                    break

                value = self.alphabeta(
                    next_game, time_left, depth - 1, alpha, beta, False)[1]

                if value > best_val:
                    best_val = value
                    best_move = move

                alpha = max(alpha, best_val)

                if beta <= alpha:
                    break
        else:
            best_val = float('inf')
            for move in game.get_legal_moves():
                next_game, _, _ = game.forecast_move(move)

                # Check time left before making a recursive call
                if time_left() <= 0:
                    break

                value = self.alphabeta(
                    next_game, time_left, depth - 1, alpha, beta, True)[1]

                if value < best_val:
                    best_val = value
                    best_move = move

                beta = min(beta, best_val)

                if beta <= alpha:
                    break

        return best_move, best_val


def depth(game, max_depth):
    if game.move_count < 10:
        return 2
    elif 10 <= game.move_count < 20:
        return 4
    elif 20 <= game.move_count < 30:
        if max_depth > 3:
            return max_depth
        else:
            return 6
    else:
        return max_depth+2
