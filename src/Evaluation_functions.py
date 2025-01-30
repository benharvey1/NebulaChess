import chess
from Board_State import State
import torch
import numpy as np
from train_MLP import MLP
from abc import ABC, abstractmethod

# TODO: Improve Classical valuator function
# TODO: Add CNN valuator function
# TODO: Simple algorithm to check blunders

class BaseValuator(ABC):
    """ Abstract base class for chess board evaluators. """

    def __init__(self):
        self.count = 0

    @abstractmethod
    def __call__(self, board):
        """Evaluates the given chess board position.
        
        Args:
            board (chess.Board): The chess board position to evaluate.

        Returns:
            float: The evaluation score of the position.
        """
        pass

    def reset(self):
        """Resets the node count."""
        self.count = 0

class MLPValuator(BaseValuator):
    """ Class to evaluate chess board positions using a trained neural network model.
    
        Attributes:
        model (Net): The neural network model for evaluation.
        device (torch.device): The device on which the model is running (CPU or GPU).
        count (int): Number of times valuator has been called since last reset
    """

    def __init__(self, path):
        """Initializes the NetValuator by loading the trained neural network model."""
        super().__init__()
        self.model = MLP()
        self.path = path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, board):
        """Evaluates the given chess board position using the neural network.
        Args:
            board (chess.Board): The current chess board position to evaluate.
        
        Returns:
            float: The evaluation score of the position.
        """
        state = torch.tensor(State(board).board_to_vector(), dtype=torch.float32).to(self.device)
        state = state.unsqueeze(0)
        score = self.model(state).item()
        self.count += 1
        return score
   
    
class ClassicValuator(BaseValuator):
    """Class to evaluate chess board position using a simple classical valuator
    
        Attributes:
        piece_values (dict): Assigns a material score to each piece type
        piece_tables (dict): Assigns score to each square for each piece type
        count (int): Number of times valuator has been called since last reset
        """
    
    def __init__(self):
        
        super().__init__()
        self.piece_values = {"P":100, "N":320, "B":330, "R":500, "Q":900, "K":20000, "K_endgame":20000,
                             "p":100, "n":320, "b":330, "r":500, "q":900, "k":20000, "k_endgame":20000}
        
        self.piece_tables = {"p": [ 0,  0,  0,  0,  0,  0,  0,  0,
                                    50, 50, 50, 50, 50, 50, 50, 50,
                                    10, 10, 20, 30, 30, 20, 10, 10,
                                    5,  5, 10, 25, 25, 10,  5,  5,
                                    0,  0,  0, 20, 20,  0,  0,  0,
                                    5, -5,-10,  0,  0,-10, -5,  5,
                                    5, 10, 10,-20,-20, 10, 10,  5,
                                    0,  0,  0,  0,  0,  0,  0,  0],

                            "P": [  0,  0,  0,  0,  0,  0,  0,  0,
                                    5, 10, 10,-20,-20, 10, 10,  5,
                                    5, -5,-10,  0,  0,-10, -5,  5,
                                    0,  0,  0, 20, 20,  0,  0,  0,
                                    5,  5, 10, 25, 25, 10,  5,  5,
                                    10, 10, 20, 30, 30, 20, 10, 10,
                                    50, 50, 50, 50, 50, 50, 50, 50,
                                    0,  0,  0,  0,  0,  0,  0,  0],
                                   
                            "n": [  -50,-40,-30,-30,-30,-30,-40,-50,
                                    -40,-20,  0,  0,  0,  0,-20,-40,
                                    -30,  0, 10, 15, 15, 10,  0,-30,
                                    -30,  5, 15, 20, 20, 15,  5,-30,
                                    -30,  0, 15, 20, 20, 15,  0,-30,
                                    -30,  5, 10, 15, 15, 10,  5,-30,
                                    -40,-20,  0,  5,  5,  0,-20,-40,
                                    -50,-40,-30,-30,-30,-30,-40,-50,],

                            "N": [  -50,-40,-30,-30,-30,-30,-40,-50,
                                    -40,-20,  0,  5,  5,  0,-20,-40,
                                    -30,  5, 10, 15, 15, 10,  5,-30,
                                    -30,  0, 15, 20, 20, 15,  0,-30,
                                    -30,  5, 15, 20, 20, 15,  5,-30,
                                    -30,  0, 10, 15, 15, 10,  0,-30,
                                    -40,-20,  0,  0,  0,  0,-20,-40,
                                    -50,-40,-30,-30,-30,-30,-40,-50,],
                                    
                            "b": [  -20,-10,-10,-10,-10,-10,-10,-20,
                                    -10,  0,  0,  0,  0,  0,  0,-10,
                                    -10,  0,  5, 10, 10,  5,  0,-10,
                                    -10,  5,  5, 10, 10,  5,  5,-10,
                                    -10,  0, 10, 10, 10, 10,  0,-10,
                                    -10, 10, 10, 10, 10, 10, 10,-10,
                                    -10,  5,  0,  0,  0,  0,  5,-10,
                                    -20,-10,-10,-10,-10,-10,-10,-20,],
                            
                            "B": [  -20,-10,-10,-10,-10,-10,-10,-20,
                                    -10,  5,  0,  0,  0,  0,  5,-10,
                                    -10, 10, 10, 10, 10, 10, 10,-10,
                                    -10,  0, 10, 10, 10, 10,  0,-10,
                                    -10,  5,  5, 10, 10,  5,  5,-10,
                                    -10,  0,  5, 10, 10,  5,  0,-10,
                                    -10,  0,  0,  0,  0,  0,  0,-10,
                                    -20,-10,-10,-10,-10,-10,-10,-20,],
                                    
                            "r": [  0,  0,  0,  0,  0,  0,  0,  0,
                                    5, 10, 10, 10, 10, 10, 10,  5,
                                    -5,  0,  0,  0,  0,  0,  0, -5,
                                    -5,  0,  0,  0,  0,  0,  0, -5,
                                    -5,  0,  0,  0,  0,  0,  0, -5,
                                    -5,  0,  0,  0,  0,  0,  0, -5,
                                    -5,  0,  0,  0,  0,  0,  0, -5,
                                    0,  0,  0,  5,  5,  0,  0,  0],

                            "R": [  0,  0,  0,  5,  5,  0,  0,  0,
                                    -5, 0,   0,  0,  0,  0,  0, -5,
                                    -5,  0,  0,  0,  0,  0,  0, -5,
                                    -5,  0,  0,  0,  0,  0,  0, -5,
                                    -5,  0,  0,  0,  0,  0,  0, -5,
                                    -5,  0,  0,  0,  0,  0,  0, -5,
                                    5,  10, 10, 10, 10, 10, 10,  5,
                                    0,  0,  0,  0,  0,  0,  0,  0],
                            
                            "q": [  0,  0,  0,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0,  0,  0,  0],

                            "Q": [  0,  0,  0,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0,  0,  0,  0,
                                    0,  0,  0,  0,  0,  0,  0,  0],

                            "k": [  -30,-40,-40,-50,-50,-40,-40,-30,
                                    -30,-40,-40,-50,-50,-40,-40,-30,
                                    -30,-40,-40,-50,-50,-40,-40,-30,
                                    -30,-40,-40,-50,-50,-40,-40,-30,
                                    -20,-30,-30,-40,-40,-30,-30,-20,
                                    -10,-20,-20,-20,-20,-20,-20,-10,
                                    20, 20,  0,  0,  0,  0, 20, 20,
                                    20, 30, 10,  0,  0, 10, 30, 20],
                                    
                            "K": [  20, 30, 10,  0,  0, 10, 30, 20,
                                    20, 20,  0,  0,  0,  0, 20, 20,
                                    -10,-20,-20,-20,-20,-20,-20,-10,
                                    -20,-30,-30,-40,-40,-30,-30,-20,
                                    -30,-40,-40,-50,-50,-40,-40,-30,
                                    -30,-40,-40,-50,-50,-40,-40,-30,
                                    -30,-40,-40,-50,-50,-40,-40,-30,
                                    -30,-40,-40,-50,-50,-40,-40,-30],
                                    
                            "k_endgame": [-50,-40,-30,-20,-20,-30,-40,-50,
                                        -30,-20,-10,  0,  0,-10,-20,-30,
                                        -30,-10, 20, 30, 30, 20,-10,-30,
                                        -30,-10, 30, 40, 40, 30,-10,-30,
                                        -30,-10, 30, 40, 40, 30,-10,-30,
                                        -30,-10, 20, 30, 30, 20,-10,-30,
                                        -30,-30,  0,  0,  0,  0,-30,-30,
                                        -50,-30,-30,-30,-30,-30,-30,-50],

                            "K_endgame": [-50,-30,-30,-30, -30,-30,-30,-50,
                                        -30,-30,  0,  0,  0,  0,-30,-30,
                                        -30,-10, 20, 30, 30, 20,-10,-30,
                                        -30,-10, 30, 40, 40, 30,-10,-30,
                                        -30,-10, 30, 40, 40, 30,-10,-30,
                                        -30,-10, 20, 30, 30, 20,-10,-30,
                                        -30,-20, -10,  0,  0,-10,-20,-30,
                                        -50,-40,-30,-20,-20,-30,-40,-50]}
        
    def is_endgame(self, board):
        """Determine if the position is in the endgame phase."""
        white_queen = len(board.pieces(chess.QUEEN, chess.WHITE))
        black_queen = len(board.pieces(chess.QUEEN, chess.BLACK))
        white_rook = len(board.pieces(chess.ROOK, chess.WHITE))
        black_rook = len(board.pieces(chess.ROOK, chess.BLACK))
        white_minor_pieces = len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.WHITE))
        black_minor_pieces = len(board.pieces(chess.KNIGHT, chess.BLACK)) + len(board.pieces(chess.BISHOP, chess.BLACK))

        no_queens = (white_queen == 0 and black_queen == 0)
        no_rooks = (white_rook == 0 and black_rook == 0)

        if no_queens:
            if (white_rook <= 1 and black_rook <= 1 and white_minor_pieces <= 3 and black_minor_pieces <= 3):
                return True  # Condition 1: No queens, ≤ 1 rook per side, ≤ 3 minor pieces
            if no_rooks:
                return True  # Condition 2: No queens, no rooks
            if white_minor_pieces <= 2 and black_minor_pieces <= 2:
                return True  # Condition 3: No queens, ≤ 2 minor pieces
        if no_rooks and white_minor_pieces <= 1 and black_minor_pieces <= 1:
            return True  # Condition 4: No rooks, ≤ 1 minor piece
        
        return False

    def mobility_score(self, board):
        """
        Calculates a mobility score based on the difference in the number of legal moves available to each side.
        
        Mobility is weighted positively for White and negatively for Black.
        """
        mobility_weight = 10
        original_turn = board.turn

        try:
            # White mobility
            board.turn = chess.WHITE
            white_mobility = len(list(board.legal_moves))

            # Black mobility
            board.turn = chess.BLACK
            black_mobility = len(list(board.legal_moves))
        finally:
            # Restore original turn
            board.turn = original_turn

        return mobility_weight * (white_mobility - black_mobility)
    
    def attackers(self, board, square, colour):
        """
        Returns set of squares for the set of attackers of the given color for the given square
        """
        return board.attackers(colour, square)
    
    def evaluate_exchanges(self, board, piece_value, attacker_squares, defender_squares):
        """
        Calculates material change from an exchange
        """

        attackers = [board.piece_at(square) for square in attacker_squares]
        attacker_values = [self.piece_values[piece.symbol()] for piece in attackers]

        defenders = [board.piece_at(square) for square in defender_squares]
        defender_values = [self.piece_values[piece.symbol()] for piece in defenders]

        
        sorted_attackers = sorted(zip(attackers, attacker_values), key=lambda x:x[1])
        sorted_defenders = sorted(zip(defenders, defender_values), key=lambda x:x[1])

        net_score = 0
        current_piece_value = piece_value

        # Simulate the exchange
        while sorted_attackers:
            # Attacker (of lowest material value) takes
            net_score -= current_piece_value
            attacker = sorted_attackers.pop(0)   # weakest attacker
            attacker_value = attacker[1]

            if not defender_values:
                break

            # Defender considers retaliation
            defender_value = sorted_defenders[0][1]  # Get the weakest defender
            if defender_value > attacker_value:
            # Defender won't capture if it results in a material loss
                break

            # Defender retaliates
            defender_values.pop(0)  # Remove this defender from the list
            net_score += attacker_value # Gain from capturing attacker

            # Update current piece value for the next iteration
            current_piece_value = self.piece_values[attacker.symbol()]

        pass

    
    def __call__(self, board):
        # TODO: Add King safety

        self.count += 1
        white_score = 0
        black_score = 0
        white_pieces = {"P", "N", "B", "R", "Q", "K"}
        black_pieces = {"p", "n", "b", "r", "q", "k"}
        endgame = self.is_endgame(board)
        for square in range(64):
            piece = board.piece_at(square)

            if piece is not None:
                piece_symbol = piece.symbol()

                if piece_symbol in white_pieces:
                    if piece_symbol == "K" and endgame:
                        piece_symbol = "K_endgame"
                    white_score += self.piece_values[piece_symbol]
                    white_score += self.piece_tables[piece_symbol][square]
                    attackers = self.attackers(board, square, chess.BLACK)
                    defenders = self.attackers(board, square, chess.WHITE)
    

                elif piece_symbol in black_pieces:
                    if piece_symbol == "k" and endgame:
                        piece_symbol = "k_endgame"
                    black_score += self.piece_values[piece_symbol]
                    black_score += self.piece_tables[piece_symbol][square]

        mobility = self.mobility_score(board)

        return (white_score - black_score) + mobility
        

if __name__ == "__main__":
    v = MLPValuator('MLP.pth')
    c = ClassicValuator()