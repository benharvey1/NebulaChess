import chess
import numpy as np
import time

# TODO: Simple algorithm to check blunders
# TODO: combine with classical valuation?
# TODO: Add mate detector part
# TODO: Can i increase speed of negamax so i can use higher search depth?
# TODO: Add a transposition table - finish this
# TODO: properly undertsand alpha-beta pruning - make notebook
# TODO: iterative deepening (time based search)
# TODO: Add dynamic time limit
# TODO: principal variation search

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

class Search():

    def __init__(self):

        self.TranspositionTable = TranspositionTable()

    def negamax(self, valuator, board, depth, colour, alpha=-float('inf'), beta=float('inf'), display_move_list = False, beam_width=10, max_depth=3):
        """Function to find the best move using the Negamax algorithm with alpha-beta pruning.
        
        Negamax is a variant of the minimax algorithm that exploits the fact that in a 
        two-player zero-sum game (like chess), the value of a game state from the perspective 
        of one player is equal to the negative of the value from the perspective of the other player.

        Alpha-beta pruning is used to optimize the search by cutting off branches that are 
        guaranteed to be worse than the current best move.

        Args:
            board: Current board position (a `chess.Board` object).
            depth (int): The number of moves to look ahead in the game tree.
            colour (int): The color of the player making the move (1 for WHITE, -1 for BLACK).
            alpha (int): The best score that the current player can guarantee so far.
            beta (int): The best score that the opponent can guarantee so far.
            display_move_list (bool): if True, function returns list of moves with valuations from 
            players perspective.
            beam_width (int): width of beam search
            max_depth (int): the maximum number of moves to look ahead.
        
        Returns:
            tuple: A tuple containing the best score, the corresponding best move and list of all moves 
            with corresponding scores. 
                The best move is a `chess.Board.move` object representing the optimal move for the player.
        """
        #board_hash = hash(board.fen())    # implement zobrist hash
        
        moves = list(board.legal_moves)  # Get all legal moves for the current player
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
        all_move_scores = [] if display_move_list else None

        for score, move in ordered_moves:
            if depth == 0:
                value = score*colour    # multiply score by colour so best value is always the largest
            
            # For each possible move, create a new board and evaluate recursively
            else:
                new_board = board.copy()  # Make a copy of the current board state
                new_board.push(move)  # Apply the current move to the copied board

                # Recursively call negamax for the new board, and negate the returned value
                # The value is negated because we are switching perspectives between the two players.
                value, _ , _ = self.negamax(valuator, new_board, depth-1, -colour, -beta, -alpha, display_move_list, beam_width, max_depth)
                value = -value

            if display_move_list:
                all_move_scores.append((move, value))
                

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
        
        if display_move_list:
            # we don't sort based on colour here since value = score * colour
            # Therefore, best move is always one with highest value (independent of colour) 
            all_move_scores.sort(key=lambda x: x[1], reverse=True)

        # If we have exact value of position we store it in the transposition table with label 'exact'
        # If position was 
          
        return max_value, best_move, all_move_scores



    def iterative_deepening(self, valuator, board, time_limit, colour, alpha=-float('inf'), beta=float('inf'), display_move_list=False, beam_width=10, max_depth=3):
        """
        Function to find the best move using the Negamax algorithm with alpha-beta pruning
        and iterative deepening, based on a time limit.
        
        Args:
            board: Current board position (a `chess.Board` object).
            time_limit (float): Maximum time allowed for computation (in seconds).
            colour (int): The color of the player making the move (1 for WHITE, -1 for BLACK).
            alpha (int): The best score that the current player can guarantee so far.
            beta (int): The best score that the opponent can guarantee so far.
            display_move_list (bool): If True, function returns a list of moves with valuations 
                                    from the player's perspective.
            beam_width (int): Width of beam search.
            max_depth (int): The maximum depth to explore.

        Returns:
            tuple: Best score, best move, and optionally the list of all move scores.
        """
        start_time = time.time()  # Record start time
        best_move = None
        best_score = -float('inf')
        all_move_scores = []

        for depth in range(0, max_depth + 1):
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit:
                break

            # Call the negamax function for the current depth
            current_score, current_best_move, current_all_scores = self.negamax(
                valuator, board, depth, colour, alpha, beta, display_move_list, beam_width, max_depth
            )

            if current_best_move:
                best_move = current_best_move
                best_score = current_score
                if display_move_list:
                    all_move_scores = current_all_scores

            # Check if time is up after processing this depth
            if time.time() - start_time >= time_limit:
                break

        return best_score, best_move, all_move_scores if display_move_list else None
    

    def move(self, valuator, board, depth, colour):
        """Get the engine's move"""

        score, best_move, all_moves = self.negamax(valuator, board, depth, colour, max_depth=depth, display_move_list=True)
        board.push(best_move)

        return best_move, all_moves
