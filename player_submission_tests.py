#!/usr/bin/env python
import traceback
from player_submission import OpenMoveEvalFn, CustomEvalFn, CustomPlayer
from game import Board, game_as_text
from test_players import RandomPlayer, HumanPlayer
import platform
if platform.system() != 'Windows':
    import resource
from time import time, sleep


def main():
    """
    print("")
    try:
        # Example test to make sure
        # your minimax works, using the
        # OpenMoveEvalFunction evaluation function.
        # This can be used for debugging your code
        # with different model Board states.
        # Especially important to check alphabeta
        # pruning

        # create dummy 5x5 board
        b = Board(RandomPlayer(), CustomPlayer(4), 5, 5)

        b.__board_state__ = [
            [" ", " " , " ", " ", " "],
            [" ", " ",  " ", " ", " "],
            [" ", " ",  " ","Q1", " "],
            [" ", " ",  " ","Q2", " "],
            [" ", " " , " ", " ", " "]
        ]
        b.__last_queen_move__[b.__queen_1__] = (2, 3, False)
        b.__last_queen_move__[b.__queen_2__] = (3, 3, False)
        b.move_count = 2

        output_b = b.copy()
        legal_moves=b.get_legal_moves()
        winner, move_history,  termination = b.play_isolation(
            time_limit=100000, print_moves=True)
        print('Minimax Test: Runs Successfully')
        # Uncomment to see example game
        #insert in reverse order
        #initial_turn = [(2, 3, False), (3, 3, False)]
        #move_history.insert(0, initial_turn)
        #print game_as_text(winner, move_history, termination, output_b)
    except NotImplementedError:
        print('Minimax Test: Not Implemented')
    except:
        print('Minimax Test: ERROR OCCURRED')
        print(traceback.format_exc())
    """

    """Example test you can run
    to make sure your AI does better
    than random or YOU!"""
    print("")
    try:
        r = RandomPlayer()  # or HumanPlayer()
        h = CustomPlayer(3)  # or HumanPlayer()
        game = Board(r, h, 7, 7)
        output_b = game.copy()
        winner, move_history, termination = game.play_isolation(
            time_limit=1000, print_moves=True)
        print("\n{} has won. Reason: {}".format(winner, termination))
        # Uncomment to see game
        # print game_as_text(winner, move_history, termination, output_b)
    except NotImplementedError:
        print('CustomPlayer Test: Not Implemented')
    except:
        print('CustomPlayer Test: ERROR OCCURRED')
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
