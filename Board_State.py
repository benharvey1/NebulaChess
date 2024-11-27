import numpy as np
import chess


class State():
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def board_to_tensor(self):
        """
        Converts current board state into a 5-layer tensor.

        Input:
            - object from board class (chess.Board())

        Output:
            - Returns a 5x8x8 numpy array representing the board:
                * Layers 0-3: Encoded 4-bit representation of pieces on the board.
                * Layer 4: Indicates player turn (0 = white, 1 = black).
        """
        

        # check board is valid
        assert self.board.is_valid()

        state_tensor = np.zeros((5, 8, 8), dtype=np.uint8)
        board_tensor = np.zeros(64, dtype=np.uint8)

        # White pieces numbered from 1-6. 7 represents white rooks when castling available.
        # Black pieces numbered from 9-14. 15 represents black rooks when castling available.
        # 8 represents possible en passant squares
        pieces = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, "p": 9, "n":10, "b":11, "r":12, "q":13, "k": 14}

        # Loop over all squares on board
        for i in range(64):
            piece = self.board.piece_at(i)   # returns piece on each square. None if square empty

            if piece is not None:
                board_tensor[i] = pieces[piece.symbol()]

        # Check castling rights
        if self.board.has_kingside_castling_rights(chess.WHITE):
            board_tensor[7] = 7
        if self.board.has_queenside_castling_rights(chess.WHITE):
            board_tensor[0] = 7
        if self.board.has_kingside_castling_rights(chess.BLACK):
            board_tensor[63] = 15
        if self.board.has_queenside_castling_rights(chess.BLACK):
            board_tensor[56] = 15

        # Check en passant 
        if self.board.ep_square is not None:
            board_tensor[self.board.ep_square] = 8

        board_tensor = board_tensor.reshape(8,8)

        # Convert each number to binary and split bits between first 4 layers of state_tensor
        # Each number 0-15 can be represented using 4-bits (0: 0000, 1: 0001, 2: 0010, 3: 0011, ... , 14: 1110, 15: 1111)
        # First layer contains the first bits, second layer contains second bits etc.

        # x >> y returns x with bits shifted to the right by y place. &1 means we only consider rightmost bit
        state_tensor[0] = (board_tensor >> 3) & 1
        state_tensor[1] = (board_tensor >> 2) & 1
        state_tensor[2] = (board_tensor >> 1) & 1
        state_tensor[3] = (board_tensor >> 0) & 1

        # Final layer denotes whose turn it is (1 = white, 0 = black)
        state_tensor[4] = self.board.turn*1.0

        return state_tensor
    
    def legal_moves(self):
        """Returns list of the legal moves for current board state"""
        return list(self.board.legal_moves)



if __name__ == "__main__":
    s = State()