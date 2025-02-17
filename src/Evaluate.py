import chess
from Board_State import State
import torch
import numpy as np
from train_MLP import MLP
from train_cnn import ResNet
from abc import ABC, abstractmethod


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
        single_input = False
        if isinstance(boards, chess.Board):
            boards = [boards]
            single_input = True

        with torch.no_grad():
            states = [State(board).board_to_vector() for board in boards]
            state_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
            scores = self.model(state_tensor).cpu().numpy()
        
        self.count += scores.shape[0]
        
        return scores.flatten()[0] if single_input else scores.flatten().tolist()
    
class CNNValuator(BaseValuator):
    """ Class to evaluate chess board positions using a trained neural network model.
    
        Attributes:
        model (Net): The neural network model for evaluation.
        device (torch.device): The device on which the model is running (CPU or GPU).
        count (int): Number of times valuator has been called since last reset
    """

    def __init__(self, path):
        """Initializes the NetValuator by loading the trained neural network model."""
        super().__init__()
        self.model = ResNet(num_residual_blocks=6)
        self.path = path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, boards):
        """Evaluates the given chess board position using the neural network.
        Args:
            board (chess.Board): The current chess board position to evaluate.
        
        Returns:
            float: The evaluation score of the position.
        """
        single_input = False
        if isinstance(boards, chess.Board):
            boards = [boards]
            single_input = True

        with torch.no_grad():
            states = [State(board).board_to_tensor() for board in boards]
            state_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
            scores = self.model(state_tensor).cpu().numpy()
        
        self.count += scores.shape[0]
        
        return scores.flatten()[0] if single_input else scores.flatten().tolist()
   

        