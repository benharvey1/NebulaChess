import chess
import chess.engine
from Evaluation_functions import MLPValuator, ClassicValuator
from search_v1 import Searchv1
from search_v2 import Searchv2
from search_v3 import Searchv3
from play import Engine, print_unicode_board, time_for_move
from tqdm import tqdm
import time
import os
import pandas as pd


def self_play(engine_1, engine_2, engine_time, increment, starting_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', show_board=False):
    """Main Loop for self play between two engines"""
    
    board = chess.Board(starting_fen)
    engine_1_colour = chess.WHITE
    engine_2_colour = chess.BLACK
    engine_1_time = engine_time
    engine_2_time = engine_time

    if show_board:
        print_unicode_board(board, perspective=engine_1_colour)

    while not board.is_game_over():

        if board.turn == engine_1_colour:
            time_limit = time_for_move(increment, engine_1_time, engine_1.number_moves)
            start_time = time.time()
            engine_1.move(board, 2*int(engine_1_colour)-1, time_limit, print_statements=False)
            end_time = time.time()
            elapsed_time = end_time - start_time
            engine_1_time -= elapsed_time
            engine_1_time += increment

        else:
            time_limit = time_for_move(increment, engine_2_time, engine_2.number_moves)
            start_time = time.time()
            engine_2.move(board, 2*int(engine_2_colour)-1, time_limit, print_statements=False)
            end_time = time.time()
            elapsed_time = end_time - start_time
            engine_2_time -= elapsed_time
            engine_2_time += increment
            
        if show_board:
            print_unicode_board(board, engine_1_colour)

    if show_board: 
        print_unicode_board(board, engine_1_colour)

    if board.is_checkmate():
        return 1 if board.turn == engine_2_colour else -1 

    return 0 


def play_match(engine1, engine2, fen, time_per_game, increment):
    """Plays two games for a given FEN, swapping colors."""

    result1 = self_play(engine1, engine2, time_per_game, increment, starting_fen=fen, show_board=False)
    result2 = self_play(engine2, engine1, time_per_game, increment, starting_fen=fen, show_board=False)

    return result1, result2

fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", # starting
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1", # Ruy Lopez
        "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", # Giuoco Piano Game
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 1", # 4 knights Game
        "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 1" # Queen's Gambit declined: Queen's Knight Variation
        "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1", # Scicilian Defense
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", # Italian Game: Two Knights Defense
        "rnbqkb1r/pppppp1p/5np1/8/3P1B2/2N5/PPP1PPPP/R2QKBNR b KQkq - 0 1", # Indian Game
        "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 1", # Slav Defense: Modern Line
        "rnbqkb1r/ppp1pppp/5n2/8/2pP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 1", # Queens's Gambit accepted
        ]

if __name__ == "__main__":

    valuator_1_path = 'models/MLP_v1.pth'
    valuator_2_path = 'models/MLP_v6.pth'

    e1 = Engine(MLPValuator(valuator_1_path), Searchv1())
    e2 = Engine(MLPValuator(valuator_2_path), Searchv2())

    engine1_name = f"{os.path.splitext(os.path.basename(valuator_1_path))[0]} + {e1.search.__class__.__name__}"
    engine2_name = f"{os.path.splitext(os.path.basename(valuator_2_path))[0]} + {e2.search.__class__.__name__}"

    time_per_game = 60
    increment = 1

    results = {"engine1_wins": 0, "engine2_wins": 0, "draws": 0}

    with tqdm(total=len(fens) * 2, desc='Playing Games', unit='Game') as pbar:

        for fen in fens:

            res1, res2 = play_match(e1, e2, fen, time_per_game, increment)

            if res1 == 1:
                results["engine1_wins"] += 1
            elif res1 == -1:
                results["engine2_wins"] += 1
            else:
                results["draws"] += 1

            if res2 == 1:
                results["engine2_wins"] += 1
            elif res2 == -1:
                results["engine1_wins"] += 1
            else:
                results["draws"] += 1

            pbar.set_postfix(
                e1_wins=results["engine1_wins"], 
                e2_wins=results["engine2_wins"], 
                draws=results["draws"]
            )
            pbar.update(2)

    new_row = {
    "Engine 1 Name": engine1_name,
    "Engine 1 Wins": results["engine1_wins"],
    "Draws": results["draws"],
    "Engine 2 Wins": results["engine2_wins"],
    "Engine 2 Name": engine2_name}

    new_row_df = pd.DataFrame([new_row])

    csv_filename = "Data/engine_comparison_results.csv"

    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        df = pd.concat([df, new_row_df], ignore_index=True)
    else:
        df = new_row_df

    print(f"Results saved to '{csv_filename}'")
    print(f"\nFinal Tally:\n{results}")


    