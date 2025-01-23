#!/usr/bin/env python
from game import Board, game_as_text
from random import randint
import game


# This file is your main submission that will be graded against. Do not
# add any classes or functions to this file that are not part of the classes
# that we want.


class OpenMoveEvalFn:

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state

        Evaluation function that outputs a score equal to how many
        moves are open for AI player on the board minus how many moves
        are open for Opponent's player on the board.
        Note:
            1. Be very careful while doing opponent's moves. You might end up
               reducing your own moves.
            3. If you think of better evaluation function, do it in CustomEvalFn below.

            Args
                param1 (Board): The board and game state.
                param2 (bool): True if maximizing player is active.

            Returns:
                float: The current state's score. MyMoves-OppMoves.

            """

        # TODO: finish this function!
        return len(game.get_legal_moves()) - len(game.get_opponent_moves())

        raise NotImplementedError


class CustomEvalFn:
    def __init__(self):
        pass

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state

        Custom evaluation function that acts however you think it should. This
        is not required but highly encouraged if you want to build the best
        AI possible.

        Args
            game (Board): The board and game state.
            maximizing_player_turn (bool): True if maximizing player is active.

        Returns:
            float: The current state's score, based on your own heuristic.

        """
        my_moves = len(game.get_legal_moves())
        opponent_moves = len(game.get_opponent_moves())
        mobility_score = my_moves - opponent_moves
        center_control_score = self.evaluate_center(game)
        threat_score = self.evaluate_threats(game)

        total_score = mobility_score + center_control_score + threat_score

        return total_score if maximizing_player_turn else -total_score
        # TODO: finish this function!
        raise NotImplementedError

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
                score -= 10
        return score


class CustomPlayer:
    # TODO: finish this class!
    """Player that chooses a move using your evaluation function
    and a minimax algorithm with alpha-beta pruning.
    You must finish and test this player to make sure it properly
    uses minimax and alpha-beta to return a good move."""

    def __init__(self, search_depth, eval_fn=CustomEvalFn()):
        """Initializes your player.

        if you find yourself with a superior eval function, update the default
        value of `eval_fn` to `CustomEvalFn()`

        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Utility function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth

    def move(self, game, legal_moves, time_left):
        """Called to determine one move by your agent

            Note:
                1. Do NOT change the name of this 'move' function. We are going to call
                the this function directly.
                2. Change the name of minimax function to alphabeta function when
                required. Here we are talking about 'minimax' function call,
                NOT 'move' function name.
                Args:
                game (Board): The board and game state.
                legal_moves (list): List of legal moves
                time_left (function): Used to determine time left before timeout

            Returns:
                tuple: best_move
            """

        best_move, utility = self.alphabeta(
            game, time_left, depth=self.search_depth)
        return best_move

    def utility(self, game, maximizing_player):
        """Can be updated if desired. Not compulsory. """
        return self.eval_fn.score(game)

    def minimax(self, game, time_left, depth, maximizing_player=True):
        """Implementation of the minimax algorithm

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, val
        """
        # TODO: finish this function!
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

    # Maximizing player strategy

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
        """Implementation of the alphabeta algorithm

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, val
        """
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
        # TODO: finish this function!
        # raise NotImplementedError
