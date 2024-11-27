import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chess
import random
import chess.engine
from tqdm import tqdm
import asyncio

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

stockfish_path = 'stockfish\\stockfish-windows-x86-64-sse41-popcnt.exe'
dataset = pd.read_csv('chessData.csv').drop_duplicates()
print(f"original dataset size: {len(dataset)}")

def add_random_move(fen):
    """Generates a random move and updates the board."""
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if legal_moves:
        random_move = random.choice(legal_moves)
        board.push(random_move)
    return board

def evaluate_with_stockfish(engine, board, depth):
    """Evaluates the position on the board using Stockfish."""
    result = engine.analyse(board, chess.engine.Limit(depth=depth))
    evaluation = result["score"].white().score()
    return evaluation

fen_strings = dataset['FEN'].to_numpy()
eval_strings = dataset['Evaluation'].to_numpy()
evals = np.zeros(len(eval_strings), dtype=np.float32)
drawing_games = []

for i in range(len(eval_strings)):
    eval = eval_strings[i]
    fen = fen_strings[i]
    if eval.startswith('#'):
        evals[i] = -10000.0 if '-' in eval else 10000.0
    else:
        evals[i] = float(eval)
        if -50 < float(eval) < 50:
            drawing_games.append(fen)

fen_strings_expanded = np.zeros(len(drawing_games), dtype=object)
evals_expanded = np.zeros(len(drawing_games), dtype=np.float32)

depth = 10
with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
    for i in tqdm(range(len(drawing_games)), desc='Expanding dataset', unit='Board'):
        fen = drawing_games[i]
        new_board = add_random_move(fen)
        eval = evaluate_with_stockfish(engine, new_board, depth)
        fen_strings_expanded[i] = new_board.fen()
        evals_expanded[i] = eval

total_fens = np.concatenate((fen_strings, fen_strings_expanded))
total_evals = np.concatenate((evals, evals_expanded))

print(f"size of artifical dataset: {len(fen_strings_expanded)}")
print(f"size of full dataset after expansion: {len(total_fens)}")

expanded_dataset = pd.DataFrame({'FEN': total_fens, 'Evaluation': total_evals}).drop_duplicates()
expanded_dataset.to_csv('ExpandedData.csv')

transformed_evals = []
transformed_fens = []

for i in range(len(total_fens)):
    fen = total_fens[i]
    eval = total_evals[i]

    if -5000 <= eval <= 5000:
        transformed_fens.append(fen)
        transformed_evals.append(eval/5000)

print(f"size of transformed dataset: {len(transformed_fens)}")
transformed_dataset = pd.DataFrame({'FEN': transformed_fens, 'Evaluation': transformed_evals})
transformed_dataset.to_csv('TransformedData.csv')