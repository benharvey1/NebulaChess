import pandas as pd
from Board_State import State
import torch
from torch.utils.data import Dataset

class ChessDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (str): Path to the CSV file containing FEN strings and evaluations.
        """
        self.data = pd.read_csv(csv_file)
        self.state_converter = State()

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (board_tensor, evaluation) where:
                - board_tensor (torch.Tensor): The 5x8x8 board representation.
                - evaluation (float): The evaluation value.
        """
        # Extract FEN string and evaluation
        fen = self.data.iloc[idx]['FEN']
        evaluation = self.data.iloc[idx]['Evaluation']

        # Generate board tensor
        self.state_converter.board.set_fen(fen)
        board_tensor = self.state_converter.board_to_tensor()

        # Convert evaluation
        if isinstance(evaluation, str) and evaluation.startswith('#'):
            # Mate evaluations: '#' indicates mate, positive for white and negative for black
            eval_value = 10000.0 if '+' in evaluation else -10000.0
        else:
            eval_value = float(evaluation)

        # Convert to PyTorch tensors
        board_tensor = torch.tensor(board_tensor, dtype=torch.float32)
        eval_value = torch.tensor(eval_value, dtype=torch.float32)

        return board_tensor, eval_value




