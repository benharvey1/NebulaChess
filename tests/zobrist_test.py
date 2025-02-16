import chess
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import zobrist

zobirst = zobrist.ZobristHash()
incorrect_hashes = 0

file_path = os.path.join(PROJECT_ROOT, "tests/zobrist_test_positions.txt")

with open(file_path, "r") as f:
    lines = f.readlines()

for line in lines:
    fen, move_string = line.strip().split(',')
    move = chess.Move.from_uci(move_string)
    board = chess.Board(fen)
    board_hash = zobirst.hash_board(board)
    
    updated_hash = zobirst.update_hash(board_hash, board, move)
    
    new_board = board.copy()
    new_board.push(move)
    hash_from_scratch = zobirst.hash_board(new_board)

    if updated_hash != hash_from_scratch:
        print("Error: Hashes do not match")
        print(fen, move)
        incorrect_hashes += 1

print(f"Incorrect hashes: {incorrect_hashes}")
