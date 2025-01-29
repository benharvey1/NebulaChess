import chess
import numpy as np
import time
import random
from src.base_search import BaseSearch

# TODO: Add fixed size to transposition table - how to replace keys
# TODO: principal variation search
# TODO: Update Zobrist hash in efficient way (using xor)
# TODO: Iterative deepending - revert back to previous depth or use current best move?

class ZobristHash():

    def __init__(self):
        self.array = [random.getrandbits(64) for _ in range(781)]

    def hash_board(self, board):

        zobrist_hash = 0

        # piece positions
        for square in range(64):
            piece = board.piece_at(square)
            
            if piece is not None:
                piece_index = (piece.piece_type - 1)*2 + int(piece.color)
                zobrist_hash ^= self.array[64*piece_index + square]

        # castling
        if board.has_kingside_castling_rights(chess.WHITE):
            zobrist_hash ^= self.array[768]
        if board.has_queenside_castling_rights(chess.WHITE):
            zobrist_hash ^= self.array[768+1]
        if board.has_kingside_castling_rights(chess.BLACK):
            zobrist_hash ^= self.array[768+2]
        if board.has_queenside_castling_rights(chess.BLACK):
            zobrist_hash ^= self.array[768+3]

        # en-passant
        if board.ep_square:
            # But only if there's actually a pawn ready to capture it. Legality
            # of the potential capture is irrelevant.
            if board.turn == chess.WHITE:
                ep_mask = chess.shift_down(chess.BB_SQUARES[board.ep_square])
            else:
                ep_mask = chess.shift_up(chess.BB_SQUARES[board.ep_square])
            ep_mask = chess.shift_left(ep_mask) | chess.shift_right(ep_mask)

            if ep_mask & board.pawns & board.occupied_co[board.turn]:
                zobrist_hash^= self.array[772 + chess.square_file(board.ep_square)]

        # turn
        if board.turn == chess.WHITE:
            zobrist_hash ^= self.array[780]

        return zobrist_hash


class TranspositionTable():

    def __init__(self):
        self.table = {}

    def store(self, zobrist_hash, depth, value, flag):
        
        entry = self.table.get(zobrist_hash)
        if entry is None or entry['depth'] <= depth:
            self.table[zobrist_hash] = {'depth': depth, 'value': value, 'flag': flag}


    def lookup(self, zobrist_hash):
        return self.table.get(zobrist_hash, None)


class Searchv3(BaseSearch):

    def __init__(self):

        super().__init__()
        self.TranspositionTable = TranspositionTable()
        self.zobrist = ZobristHash()

    def root_negamax(self, start_time, time_limit, valuator, board, depth, colour):

        """Wrapper function to find the best move.

        Args:
            start_time: Time at which negamax is called
            time_limit: time to find move
            valuator: evaluation function instance
            board: Current board position (a `chess.Board` object).
            depth (int): The number of moves to look ahead in the game tree.
            colour (int): The color of the player making the move (1 for WHITE, -1 for BLACK).
        
        Returns:
            the best move
        """

        best_move = None
        max_value = -float('inf')

        # Move ordering
        moves = list(board.legal_moves)
        scores = []
        
        for move in moves:
            new_board = board.copy()
            new_board.push(move)

            if new_board.is_checkmate():
                scores.append(-0.999 if colour == 1 else 0.999)
            
            elif new_board.is_stalemate() or new_board.is_insufficient_material() or new_board.is_seventyfive_moves():
                scores.append(0)
            
            else:
                # would be better to simply update the hashes using xor function
                temp_zobrist_hash = self.zobrist.hash_board(new_board)
                entry = self.TranspositionTable.lookup(temp_zobrist_hash)

                if entry is not None and entry['flag'] == 'EXACT':
                    # This is just a shallow search used for move ordering so we do not have to worry about depth
                    # Maybe only for EXACT nodes - not sure
                    scores.append(entry['value'])
                else:
                    scores.append(valuator(new_board))

        ordered_moves = sorted(zip(scores, moves), key=lambda x: x[0], reverse=(colour==1))

        for _, move in ordered_moves:

            if time.time() - start_time >= time_limit:
                return None

            new_board = board.copy()
            new_board.push(move)

            result = self.negamax(start_time, time_limit, valuator, new_board, depth-1, -colour, alpha=-float('inf'), beta=float('inf'), beam_width=10, max_depth=depth)
            if result is None:
                return None
            
            value = -result

            if value > max_value:
                max_value = value
                best_move = move

        return best_move
    
    def negamax(self, start_time, time_limit, valuator, board, depth, colour, alpha=-float('inf'), beta=float('inf'), beam_width=10, max_depth=3):

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
            return None
        
        # Check in transposition table
        zobrist_hash = self.zobrist.hash_board(board)    # implement zobrist hash
        entry = self.TranspositionTable.lookup(zobrist_hash)

        if entry is not None and entry['depth'] >= depth:

            if entry['flag'] == 'EXACT':
                return entry['value']
            
            elif entry['flag'] == 'LOWER' and entry['value'] >= beta:
                return entry['value']
                
            elif entry['flag'] == 'UPPER' and entry['value'] <= alpha:
                return entry['value']

        # base case 
        if depth == 0:

            value = valuator(board)*colour

            # add to transposition table
            self.TranspositionTable.store(zobrist_hash, depth, value, 'EXACT')

            return value
        


        # Move ordering
        moves = list(board.legal_moves)
        scores = []
        
        for move in moves:
            new_board = board.copy()
            new_board.push(move)

            # Check if it's a checkmate, stalemate, insufficient material, or draw condition
            if new_board.is_checkmate():
                scores.append(-0.999 if colour == 1 else 0.999)
            
            elif new_board.is_stalemate() or new_board.is_insufficient_material() or new_board.is_seventyfive_moves():
                scores.append(0)  # Draw scenario
            
            else:
                # would be better to simply update the hashes using xor function
                temp_zobrist_hash = self.zobrist.hash_board(new_board)
                entry = self.TranspositionTable.lookup(temp_zobrist_hash)

                if entry is not None and entry['flag'] == 'EXACT':
                    # This is just a shallow search used for move ordering so we do not have to worry about depth
                    # Maybe only for EXACT nodes - not sure
                    scores.append(entry['value'])
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
        if remaining_depth >= 2:
            effective_beam_width = min(beam_width, len(ordered_moves))
            ordered_moves = ordered_moves[:effective_beam_width]


        # Recursive Loop    
        for score, move in ordered_moves:

            flag = 'UPPER'

            new_board = board.copy()  # Make a copy of the current board state
            new_board.push(move)  # Apply the current move to the copied board

            # Recursively call negamax for the new board, and negate the returned value
            # The value is negated because we are switching perspectives between the two players.
            result = self.negamax(start_time, time_limit, valuator, new_board, depth-1, -colour, -beta, -alpha, beam_width, max_depth)

            if result is None:
                return None

            value = -result

            # If value >= beta can prune
            if value >= beta:
                self.TranspositionTable.store(zobrist_hash, depth, beta, 'LOWER')
                return beta

            # Update alpha to the maximum of the current alpha and the new value
            # Alpha is the highest value player can guarentee so far in the search
            if value > alpha:
                alpha = value
                flag = 'EXACT'

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

        for depth in range(1, maximum_depth + 1):
            #print(f'depth: {depth}, time: {elapsed_time}')
            if time.time() - start_time >= time_limit:
                break

            # Call the negamax function for the current depth
            result = self.root_negamax(
                start_time, time_limit, valuator, board, depth, colour)

            if result is None:
                break
            
            best_move = result

            # Check if time is up after processing this depth
            if time.time() - start_time >= time_limit:
                break
        
        return best_move
    

    def move(self, valuator, board, colour, time_limit):
        """Get the engine's move"""

        best_move= self.iterative_deepening(valuator, board, time_limit, colour, maximum_depth=5)
        #print(len(self.TranspositionTable.table))

        return best_move

