import chess
import numpy as np
import time
import random
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import base_search 
import zobrist

# Version 2 of the Search
# Implemented Transposition table with Zobrist Hashing

class TranspositionTable():

    def __init__(self):
        self.table = {}

    def store(self, zobrist_hash, depth, value, best_move, flag):

        self.table[zobrist_hash] = {'depth': depth, 'value': value, 'best_move': best_move, 'flag': flag}

    def lookup(self, zobrist_hash, depth):

        if zobrist_hash in self.table:
            entry = self.table[zobrist_hash]
            if entry['depth'] >= depth:
                return entry
        return None

class Searchv2(base_search.BaseSearch):

    def __init__(self):

        super().__init__()
        self.TranspositionTable = TranspositionTable()
        self.zobrist = zobrist.ZobristHash()

    def negamax(self, start_time, time_limit, valuator, board, depth, colour, alpha=-float('inf'), beta=float('inf'), beam_width=10, max_depth=3):
        """Function to find the best move using the Negamax algorithm with alpha-beta pruning.
        
        Negamax is a variant of the minimax algorithm that exploits the fact that in a 
        two-player zero-sum game (like chess), the value of a game state from the perspective 
        of one player is equal to the negative of the value from the perspective of the other player.

        Alpha-beta pruning is used to optimize the search by cutting off branches that are 
        guaranteed to be worse than the current best move.

        Args:
            start_time: Time at which negamax is called
            time_limit: time to find move
            valuator: evaluation function instance
            board: Current board position (a `chess.Board` object).
            depth (int): The number of moves to look ahead in the game tree.
            colour (int): The color of the player making the move (1 for WHITE, -1 for BLACK).
            alpha (int): The best score that the current player can guarantee so far.
            beta (int): The best score that the opponent can guarantee so far.
            beam_width (int): width of beam search
            max_depth (int): the maximum number of moves to look ahead.
        
        Returns:
            tuple: A tuple containing the best score and the corresponding best move. 
                The best move is a `chess.Board.move` object representing the optimal move for the player.
        """
        if time.time() - start_time >= time_limit:
            return None
        
        zobrist_hash = self.zobrist.hash_board(board)    # implement zobrist hash
        entry = self.TranspositionTable.lookup(zobrist_hash, depth)

        if entry is not None:
            if entry['flag'] == 'EXACT':
                return entry['value'], entry['best_move']
            
            elif entry['flag'] == 'LOWER':
                # value is a lower bound so know we can achieve a score of at least this
                alpha = max(alpha, entry['value'])

            elif entry['flag'] == 'UPPER':
                beta = min(beta, entry['value'])
            
            if alpha >= beta:
                return entry['value'], entry['best_move']


        moves = list(board.legal_moves)  # Get all legal moves for the current player
        scores = []

        for move in moves:
            new_board = board.copy()
            new_board.push(move)
            
            # Check if it's a checkmate, stalemate, insufficient material, or draw condition
            if new_board.is_checkmate():
                scores.append(colour)
            
            elif new_board.is_stalemate() or new_board.is_insufficient_material() or new_board.is_seventyfive_moves():
                scores.append(0)  # Draw scenario
            
            else:
                # Otherwise, evaluate the board using the evaluator
                scores.append(valuator(new_board))

        # Sort moves in order of decreasing/increasing score for white/black to speed up pruning
        # i.e. search starts with best move for depth=0
        ordered_moves = sorted(zip(scores, moves), key=lambda x: x[0], reverse=(colour==1))

        # apply beam search to speed up computation
        # Stockfish top move in the top 10 moves at depth=0 around 80-90% of the time
        #effective_beam_width = min(beam_width, len(ordered_moves))
        remaining_depth = max_depth - depth
        if remaining_depth >= 1:
            effective_beam_width = min(beam_width, len(ordered_moves))
            ordered_moves = ordered_moves[:effective_beam_width]


        max_value = -float('inf')  # Start with the worst possible value for the maximizing player
        best_move = None  # Placeholder for the best move

        for score, move in ordered_moves:

            if time.time() - start_time >= time_limit:
                return None
        
            if depth == 0:
                value = score*colour    # multiply score by colour so best value is always the largest
            
            # For each possible move, create a new board and evaluate recursively
            else:
                new_board = board.copy()  # Make a copy of the current board state
                new_board.push(move)  # Apply the current move to the copied board

                # Recursively call negamax for the new board, and negate the returned value
                # The value is negated because we are switching perspectives between the two players.
                result = self.negamax(start_time, time_limit, valuator, new_board, depth-1, -colour, -beta, -alpha, beam_width, max_depth)
    
                if result is None:
                    return None
                
                value, _ = result
                value = -value
                

            # Update the highest value and the best move if a better value is found
            if value > max_value:
                max_value = value
                best_move = move
            
            # Update alpha to the maximum of the current alpha and the new value
            # Alpha is the highest value player can guarentee so far in the search
            alpha = max(alpha, value)

            # Alpha-Beta Pruning
            # Beta is the lowest value (from player's perspective) opponent can guarentee so far in search
            # If alpha >= beta, we can stop searching further down this branch 
            if alpha >= beta:
                break
        
        # flag = 'EXACT' if position is evaluated exactly (i.e. alpha < beta).
  
        # flag = 'LOWER' if value >= beta. Indicates that the search found something that was "too good".
        # Opponent has some way, already found by the search, of avoiding this position, so you have to assume that they'll do this.
        # value is a lower bound on exact value of node.

        # flag = 'UPPER' if value <= alpha. Position was not good enough for us.
        # value is an upper bound on exact value of node.

        flag = None

        if max_value >= beta:
            flag = 'LOWER'

        elif max_value <= alpha:
            flag = 'UPPER'

        else:
            flag = 'EXACT'

        self.TranspositionTable.store(zobrist_hash, depth, max_value, best_move, flag)
          
        return max_value, best_move



    def iterative_deepening(self, valuator, board, time_limit, colour, alpha=-float('inf'), beta=float('inf'), beam_width=10, maximum_depth=5):
        """
        Function to find the best move using the Negamax algorithm with alpha-beta pruning
        and iterative deepening, based on a time limit.
        
        Args:
            valuator: Evaluation function instance.
            board: Current board position (a `chess.Board` object).
            time_limit (float): Maximum time allowed for computation (in seconds).
            colour (int): The color of the player making the move (1 for WHITE, -1 for BLACK).
            alpha (int): The best score that the current player can guarantee so far.
            beta (int): The best score that the opponent can guarantee so far.
            beam_width (int): Width of beam search.
            max_depth (int): The maximum depth to explore.

        Returns:
            tuple: Best score, best move.
        """
        start_time = time.time()  # Record start time
        best_move = None

        for depth in range(0, maximum_depth + 1):
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit:
                break

            # Call the negamax function for the current depth
            result = self.negamax( start_time, time_limit,
                valuator, board, depth, colour, alpha, beta, beam_width, max_depth=depth
            )

            if result is None:
                break

            current_score, current_best_move = result

            if current_best_move:
                best_move = current_best_move
                
            # Check if time is up after processing this depth
            if time.time() - start_time >= time_limit:
                break

        if best_move is None:
            best_move = random.choice(list(board.legal_moves))

        return best_move
    

    def move(self, valuator, board, colour, time_limit):
        """Get the engine's move"""

        best_move = self.iterative_deepening(valuator, board, time_limit, colour, maximum_depth=4)
        board.push(best_move)

        return best_move