import numpy as np
import chess


class State():
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def board_to_vector(self):
        """
        Converts current board state into a flat vector of length 901.

        Input:
            - object from board class (chess.Board())

        Output:
            - Returns a (901,) numpy array representing the board
        """

        board_vector = np.zeros(901, dtype=np.uint8)

        # check board is valid
        assert self.board.is_valid()

        # Each piece type assigned a row
        pieces = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5, "p": 6, "n":7, "b":8, "r":9, "q":10, "k": 11}

        # Each row (of first 12 rows) of matrix corresponds to particular piece type
        # Each column represents a square on the board
        # Second last row shows all squares white pieces can move to
        # Last row shows all squares black pieces can move to 
        board_matrix = np.zeros((14, 64), dtype=np.uint8)

        for i in range(64):
            piece = self.board.piece_at(i)

            if piece is not None:
                row = pieces[piece.symbol()]
                board_matrix[row][i] = 1

        turn = self.board.turn

        self.board.turn = chess.WHITE
        for move in self.board.legal_moves:
            board_matrix[12][move.to_square] = 1

        self.board.turn = chess.BLACK
        for move in self.board.legal_moves:
            board_matrix[13][move.to_square] = 1

        board_vector[:896] = board_matrix.flatten() # (14,64) -> (896,)

        # Check castling rights 
        if self.board.has_kingside_castling_rights(chess.WHITE):
            board_vector[896] = 1
        if self.board.has_queenside_castling_rights(chess.WHITE):
            board_vector[897] = 1
        if self.board.has_kingside_castling_rights(chess.BLACK):
            board_vector[898] = 1
        if self.board.has_queenside_castling_rights(chess.BLACK):
            board_vector[899] = 1

        # Bit to denote whose turn (0=white, 1=black)
        board_vector[900] = turn*1.0

        return board_vector


    def board_to_tensor(self):
        """
        Converts current board state into a 19-layer tensor.

        Input:
            - object from board class (chess.Board())

        Output:
            - Returns a 19x8x8 numpy array representing the board:
                * Layers 0-11: Encoded positions of all pieces.
                * Layers 12-13: Encode possible moves
                * Layers 14-17: Indicates castling rightd
                * Layer 18: Indicates player turn (0 = white, 1 = black).
        """

        board_tensor = np.zeros((19, 8, 8), dtype=np.uint8)

        # check board is valid
        assert self.board.is_valid()

        # Each piece type assigned a row
        pieces = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5, "p": 6, "n":7, "b":8, "r":9, "q":10, "k": 11}

        # Each row (of first 12 rows) of matrix corresponds to particular piece type
        # Each column represents a square on the board
        # Second last row shows all squares white pieces can move to
        # Last row shows all squares black pieces can move to 
        board_matrix = np.zeros((14, 64), dtype=np.uint8)

        for i in range(64):
            piece = self.board.piece_at(i)

            if piece is not None:
                row = pieces[piece.symbol()]
                board_matrix[row][i] = 1

        turn = self.board.turn

        self.board.turn = chess.WHITE
        for move in self.board.legal_moves:
            board_matrix[12][move.to_square] = 1

        self.board.turn = chess.BLACK
        for move in self.board.legal_moves:
            board_matrix[13][move.to_square] = 1
        
        
        board_matrix = board_matrix.reshape(14,8,8)
        board_tensor[:14] = board_matrix

        # Castling rights
        if self.board.has_kingside_castling_rights(chess.WHITE):
            board_tensor[14] = 1
        if self.board.has_queenside_castling_rights(chess.WHITE):
            board_tensor[15] = 1
        if self.board.has_kingside_castling_rights(chess.BLACK):
            board_tensor[16] = 1
        if self.board.has_queenside_castling_rights(chess.BLACK):
            board_tensor[17] = 1

        # Final layer denotes whose turn it is (0 = white, 1 = black)
        board_tensor[18] = turn*1.0

        return board_tensor


if __name__ == "__main__":
    s = State()