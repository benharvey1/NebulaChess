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
import TranspositionTable

# Version 4 of search
# Added batch evaluations (much quicker) and fixed size transposition table

class Searchv4(base_search.BaseSearch):

    def __init__(self):

        super().__init__()
        self.TranspositionTable = TranspositionTable.TranspositionTable()
        self.zobrist = zobrist.ZobristHash()

    def order_moves(self, board, moves, valuator, colour):
        
        """Returns a list of ordered moves with expected best move first."""

        scores = []
        positions_to_evaluate = []
        move_indices = []

        for i, move in enumerate(moves):
    
            board.push(move)

            if board.is_checkmate():
                scores.append((colour, move))

            elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
                scores.append((0, move))
                
            else:
                positions_to_evaluate.append(board.copy())
                move_indices.append(i)
                

            board.pop()

        if positions_to_evaluate:
            
            batch_scores = valuator(positions_to_evaluate)
            for idx, score in zip(move_indices, batch_scores):
                move = moves[idx]
                scores.append((score, move))

        ordered_moves = sorted(scores, key=lambda x: x[0], reverse=(colour==1))
        
        return ordered_moves

    
    def root_negamax(self, start_time, time_limit, valuator, board, depth, colour, alpha, beta, current_best_move, current_best_score):

        """Wrapper function to find the best move.

        Args:
            start_time: Time at which negamax is called
            time_limit: time to find move
            valuator: evaluation function instance
            board: Current board position (a `chess.Board` object).
            depth (int): The number of moves to look ahead in the game tree.
            colour (int): The color of the player making the move (1 for WHITE, -1 for BLACK).
            current_best_move: The current best move
            current_best_score: The current_highest_score
        
        Returns:
            the best move
        """

        best_move = current_best_move
        max_value = -float('inf')

        # Move ordering
        moves = list(board.legal_moves)
        ordered_moves = self.order_moves(board, moves, valuator, colour)

        # Make sure best move from previous search is first in list
        if best_move is not None:
            for i, (score, move) in enumerate(ordered_moves):
                if move == best_move:
                    ordered_moves.pop(i)
                    ordered_moves.insert(0, (current_best_score, best_move))
                    break

        for score, move in ordered_moves:

            if time.time() - start_time >= time_limit:
                break

            board.push(move)

            value = -self.negamax(start_time, time_limit, valuator, board, score, depth-1, -colour, -beta, -alpha, beam_width=10, max_depth=depth-1)

            board.pop()

            if value > max_value:
                max_value = value
                best_move = move

            alpha = max(max_value, alpha)

            if alpha >= beta:
                break

        return best_move, max_value
    
    def negamax(self, start_time, time_limit, valuator, board, score, depth, colour, alpha=-float('inf'), beta=float('inf'), beam_width=10, max_depth=3):

        """Function to find the highest move evaluation using the Negamax algorithm with alpha-beta pruning.
        
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
            score: evaluation of board at depth=0
            depth (int): The number of moves to look ahead in the game tree.
            colour (int): The color of the player making the move (1 for WHITE, -1 for BLACK).
            alpha (int): The best score that the current player can guarantee so far.
            beta (int): The best score that the opponent can guarantee so far.
            beam_width (int): width of beam search
            max_depth (int): the maximum number of moves to look ahead.
        
        Returns:
            the best score
        """

        if time.time() - start_time >= time_limit:
            return alpha
        
        # Check in transposition table
        zobrist_hash = self.zobrist.hash_board(board)    # implement zobrist hash
        entry = self.TranspositionTable.lookup(zobrist_hash)

        if entry is not None:
            
            stored_value, stored_depth, flag = entry
            
            if stored_depth >= depth:

                if flag == self.TranspositionTable.EXACT:
                    return stored_value
                
                elif flag == self.TranspositionTable.LOWER:
                    alpha = max(alpha, stored_value)
                    
                elif flag == self.TranspositionTable.UPPER:
                    beta = min(beta, stored_value)

                if alpha >= beta:
                    return stored_value

        # base case 
        if depth == 0:

            value = score*colour

            # add to transposition table
            self.TranspositionTable.store(zobrist_hash, depth, value, self.TranspositionTable.EXACT)

            return value
        

        # Move ordering
        moves = list(board.legal_moves)
        ordered_moves = self.order_moves(board, moves, valuator, colour)

        # apply beam search to speed up computation
        # Stockfish top move in the top 10 moves at depth=0 around 80-90% of the time
        #effective_beam_width = min(beam_width, len(ordered_moves))
        remaining_depth = max_depth - depth
        if remaining_depth >= 0:
            effective_beam_width = min(beam_width, len(ordered_moves))
            ordered_moves = ordered_moves[:effective_beam_width]


        flag = self.TranspositionTable.UPPER
        # Recursive Loop    
        for score, move in ordered_moves:

            board.push(move)

            # Recursively call negamax for the new board, and negate the returned value
            # The value is negated because we are switching perspectives between the two players.
            result = self.negamax(start_time, time_limit, valuator, board, score, depth-1, -colour, -beta, -alpha, beam_width, max_depth)

            board.pop()

            value = -result

            # If value >= beta can prune
            if value >= beta:
                self.TranspositionTable.store(zobrist_hash, depth, beta, self.TranspositionTable.LOWER)
                return beta

            # Update alpha to the maximum of the current alpha and the new value
            # Alpha is the highest value player can guarentee so far in the search
            if value > alpha:
                alpha = value
                flag = self.TranspositionTable.EXACT

        self.TranspositionTable.store(zobrist_hash, depth, alpha, flag)

        return alpha


    def iterative_deepening(self, valuator, board, time_limit, colour, maximum_depth=5):
        """
        Function to find the best move using the Negamax algorithm with alpha-beta pruning
        and iterative deepening, based on a time limit.
        
        Args:
            valuator: Evaluation function instance 
            board: Current board position (a `chess.Board` object).
            time_limit (float): Maximum time allowed for computation (in seconds).
            colour (int): The color of the player making the move (1 for WHITE, -1 for BLACK).
            max_depth (int): The maximum depth to explore.

        Returns:
            tuple: Best move
        """
        start_time = time.time()  # Record start time
        best_move = None
        best_value = None

        for depth in range(1, maximum_depth + 1):

            if time.time() - start_time >= time_limit:
                break

            # Call the negamax function for the current depth
            best_move, best_value = self.root_negamax(
                start_time, time_limit, valuator, board, depth, colour, -float('inf'), float('inf'), best_move, best_value)

            # Check if time is up after processing this depth
            if time.time() - start_time >= time_limit:
                break
            
        if best_move is None:
            best_move = random.choice(list(board.legal_moves))
        
        return best_move
    

    def move(self, valuator, board, colour, time_limit):
        """Get the engine's move"""

        best_move= self.iterative_deepening(valuator, board, time_limit, colour, maximum_depth=10)
        board.push(best_move)

        return best_move