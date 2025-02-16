import chess
import chess.engine
from tqdm import tqdm
import time
import os
import sys
import pandas as pd
from search_v1 import Searchv1
from search_v2 import Searchv2
from search_v3 import Searchv3
from search_v4 import Searchv4

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import Evaluate
import engine
import play



def self_play(engine_1, engine_2, engine_time, increment, starting_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', show_board=False):
    """Main Loop for self play between two engines"""
    
    board = chess.Board(starting_fen)
    engine_1_colour = chess.WHITE
    engine_2_colour = chess.BLACK
    engine_1_time = engine_time
    engine_2_time = engine_time
    
    if show_board:
        play.print_unicode_board(board, perspective=engine_1_colour)

    while not board.is_game_over(claim_draw=True):

        if board.turn == engine_1_colour:
            time_limit = engine_1.time_for_move(engine_1_time, increment)
            start_time = time.time()
            engine_1.move(board, 2*int(engine_1_colour)-1, time_limit)
            end_time = time.time()
            elapsed_time = end_time - start_time
            engine_1_time -= elapsed_time
            engine_1_time += increment

        else:
            time_limit = engine_2.time_for_move(engine_2_time, increment)
            start_time = time.time()
            engine_2.move(board, 2*int(engine_2_colour)-1, time_limit)
            end_time = time.time()
            elapsed_time = end_time - start_time
            engine_2_time -= elapsed_time
            engine_2_time += increment
            
        if show_board:
            play.print_unicode_board(board, engine_1_colour)

    if show_board: 
        play.print_unicode_board(board, engine_1_colour)

    engine_1.clear_table()
    engine_2.clear_table()

    if board.is_checkmate():
        return 1 if board.turn == engine_2_colour else -1 

    return 0 


def play_match(engine1, engine2, fen, time_per_game, increment):
    """Plays two games for a given FEN, swapping colors."""


    result1 = self_play(engine1, engine2, time_per_game, increment, starting_fen=fen, show_board=False)
    result2 = self_play(engine2, engine1, time_per_game, increment, starting_fen=fen, show_board=False)

    return result1, result2

def read_fens(file_path):
    fens = []
    with open(file_path, 'r') as file:
        for line in file:
            fen = line.split("#")[0].strip()
            if fen:
                fens.append(fen)
    return fens

if __name__ == "__main__":

    valuator_1_path = os.path.join(PROJECT_ROOT, "models/cnn.pth")
    valuator_2_path = os.path.join(PROJECT_ROOT, "models/cnn.pth")

    e1 = engine.Engine(Evaluate.CNNValuator(valuator_1_path), Searchv4())
    e2 = engine.Engine(Evaluate.CNNValuator(valuator_1_path), Searchv5())

    engine1_name = f"{os.path.splitext(os.path.basename(valuator_1_path))[0]} + {e1.search.__class__.__name__}"
    engine2_name = f"{os.path.splitext(os.path.basename(valuator_2_path))[0]} + {e2.search.__class__.__name__}"

    time_per_game = 1
    increment = 3

    fens = read_fens(os.path.join(PROJECT_ROOT, "tests/openings.txt"))
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

    new_row_df = pd.DataFrame([{
    "Engine 1 Name": engine1_name,
    "Engine 1 Wins": results["engine1_wins"],
    "Draws": results["draws"],
    "Engine 2 Wins": results["engine2_wins"],
    "Engine 2 Name": engine2_name}])

    csv_filename = os.path.join(PROJECT_ROOT, "tests/engine_comparison_results.csv")

    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
    else:
        df = pd.DataFrame(columns=["Engine 1 Name", "Engine 1 Wins", "Draws", "Engine 2 Wins", "Engine 2 Name"])

    df = pd.concat([df, new_row_df], ignore_index=True)
    df.to_csv(csv_filename, index=False)

    print(f"Results saved to '{csv_filename}'")
    print(f"\nFinal Tally:\n{results}")


    