import chess
import chess.svg
import numpy as np
import time
from Evaluation_functions import MLPValuator, ClassicValuator, BaseValuator
from base_search import BaseSearch
from search_v1 import Searchv1
from search_v2 import Searchv2
from search_v3 import Searchv3


class Engine():

    def __init__(self, valuator: BaseValuator , search: BaseSearch):

        self.valuator = valuator
        self.search = search
        self.number_moves = 0

    def move(self, board, colour, time_limit, print_statements=True):

        if print_statements:
            print("Engine is thinking...")

        t1 = time.time()
        best_move = self.search.move(self.valuator, board, colour, time_limit)
        board.push(best_move)
        self.number_moves += 1
        
        t2 = time.time()
        t = t2 - t1

        if print_statements:
            print(f"Explored {self.valuator.count} nodes explored in {t:.3f} seconds")
            self.valuator.reset()
            print(f"Engine played {best_move.uci()}")



def time_for_move(increment, time_remaining, number_moves):
    """Returns the time available for the engine to move. 
    Based on overall time left, game increment and number of moves."""

    estimated_moves_left = max(100 - number_moves, 10)
    return increment + time_remaining/estimated_moves_left

def get_user_move(board):
    """Get the users move"""

    while True:
        uci_str = input("Input your move in UCI format: ")
    
        try:
            move = chess.Move.from_uci(uci_str) # convert to a chess.Move object
        
            if move in list(board.legal_moves): # check if move legal
                board.push(move)    # update board
                return

            else:
                print("Illegal move. Please try again.")
    
        except ValueError:
            print("Invalid input. Please use UCI format.")


def print_unicode_board(board, perspective):
    
    """Displays board in nice way in terminal"""

    # TODO: black pawns are displayed in blue - see if can fix
    sc, ec = "\x1b[0;30;107m", "\x1b[0m"
    white_square = "\x1b[48;5;253m"  
    black_square = "\x1b[48;5;245m"

    ranks = range(8) if perspective == chess.BLACK else range(7, -1, -1)
    files = range(8) if perspective == chess.WHITE else range(7, -1, -1)

    for rank in ranks:
        line = [f"{sc} {rank+1}"]
        for file in files:
            square = chess.square(file, rank)
            piece = board.piece_at(square)

            square_colour = white_square if (rank+file)%2 == 1 else black_square
            symbol = chess.UNICODE_PIECE_SYMBOLS[piece.symbol()] if piece else " "
            line.append(square_colour + symbol)
        
        print(" " + " ".join(line) + f" {sc} {ec}")
    
    if perspective == chess.WHITE:
        print(f" {sc}   a b c d e f g h  {ec}\n")
    else:
        print(f" {sc}   h g f e d c b a  {ec}\n")



def play_game(valuator, search):
    """Main loop to play against engine"""

    # TODO: need to implement some kind of mate checker
    # for both offense and defense
    
    board = chess.Board()
    number_engine_moves = 0
    engine = Engine(valuator, search)

    # Ask the user for their preferred color
    while True:
        user_colour_str = input("What colour do you want to play as? WHITE or BLACK: ").strip().upper()
        if user_colour_str == 'WHITE':
            user_colour = chess.WHITE   # chess.WHITE = True
            engine_colour = chess.BLACK # chess.BLACK = False
            break
        elif user_colour_str == 'BLACK':
            user_colour = chess.BLACK
            engine_colour = chess.WHITE
            break
        else:
            print("Invalid input. Please type WHITE or BLACK.")

    
    engine_time = int(input("Please enter the inital time for the engine in seconds: "))
    increment = int(input("Please enter the time increment for the engine in seconds: "))

    print("\nGame Start! Here is the initial board:")
    print_unicode_board(board, perspective=user_colour)

    while not board.is_game_over():

        if engine_time <= 0:
            print("\n The engine has run out of time. You win")
            break
        

        if board.turn == user_colour:
            print("\nYour move:")
            get_user_move(board)

        else:
            print("\nEngine's move.")
            time_limit = time_for_move(increment, engine_time, engine.number_moves)
            engine_start_time = time.time()
            engine.move(board, 2*int(engine_colour)-1, time_limit)
            engine_end_time = time.time()

            elapsed_time = engine_end_time - engine_start_time
            engine_time -= elapsed_time
            engine_time += increment

            number_engine_moves += 1

        print_unicode_board(board, perspective=user_colour)

    if board.is_checkmate():
        if board.turn == user_colour:
            print("\nCheckmate! The engine wins.")
        else:
            print("\nCheckmate! You win.")

    elif board.is_stalemate():
        print("\nStalemate! It's a draw.")

    elif board.is_insufficient_material():
        print("\nDraw due to insufficient material.")

    elif board.is_fivefold_repetition():
        print("\nDraw due to repetition.")

    elif board.is_seventyfive_moves():
        print("\nDraw due to 150 total moves without pawn move or capture.")

    else:
        print("\nGame over.")

    print("\nFinal Board:")
    print_unicode_board(board, perspective=user_colour)


if __name__ == "__main__":
    v = MLPValuator('MLP_final.pth')
    s = Searchv3()
    c = ClassicValuator()
    play_game(v, s)
    









    