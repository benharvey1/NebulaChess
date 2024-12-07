import numpy as np
import torch
from torch.utils.data import Dataset

class ChessDataset(Dataset):
    def __init__(self, file_path):
        """
        Args:
            file_path (str): Path to the npz file containing state tensors and evaluations.
        """
        data = np.load(file_path)
        self.X = data['X']
        self.y = data['y']

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.y)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (board_tensor, evaluation) where:
                - X_tensor (torch.Tensor): The board representation.
                - y_tensor (torch.Tensor): The evaluation value.
        """
        

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)

        return X_tensor, y_tensor




