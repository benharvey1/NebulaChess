import numpy as np
from tqdm import tqdm
from chess_dataset import ChessDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class ResidualBlock(nn.Module):
    """Standard Residual Block for ResNet"""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x  

        x = F.relu(self.bn1(self.conv1(x))) 
        x = self.bn2(self.conv2(x))  

        x += residual  
        return F.relu(x)  


class ResNet(nn.Module):
    """Convolutional Neural Network. Architecture based on AlphaZero."""

    def __init__(self, num_residual_blocks):

        super(ResNet, self).__init__()

        # initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual Blocks
        self.residual_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(num_residual_blocks)])

        # Final convolutional layer
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(1)

        # Dense layers
        self.fc1 = nn.Linear(8*8, 128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))  #input: (batch_size, 16, 8, 8), output: (batch_size, 64, 8, 8)

        for block in self.residual_blocks:
            x = block(x)    #input: (batch_size, 64, 8, 8), output: (batch_size, 64, 8, 8)

        x = F.relu(self.bn2(self.conv2(x))) #input: (batch_size, 64, 8, 8), output: (batch_size, 1, 8, 8)
        x = torch.flatten(x, start_dim=1)   #input: (batch_size, 1, 8, 8), output: (batch_size, 64)
        x = self.dropout(F.relu(self.fc1(x))) #input: (batch_size, 64), output: (batch_size, 128)
        x = torch.tanh(self.fc4(x)) #input: (batch_size, 128), output: (batch_size, 1)

        return x

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, delta, path):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path to save the best model checkpoint.
        """
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), self.path)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), self.path)


def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    
    train_loss = 0
    model.train()

    with tqdm(dataloader, desc=f"Epoch {epoch + 1}", unit="batch") as batch_progress:
        for batch, (X, y) in enumerate(batch_progress):
            X,y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y.unsqueeze(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_progress.set_postfix(loss=loss.item())
            train_loss += loss.item()
        
        train_loss /= len(dataloader)
        wandb.log({"Train Loss": train_loss})
        print(f"Avg train loss: {train_loss:6f}")


def test_loop(dataloader, model, loss_fn):

    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.unsqueeze(-1)).item()

    test_loss /= num_batches
    wandb.log({"Test Loss": test_loss})
    print(f"Avg test loss: {test_loss:6f}")
    return test_loss

if __name__ == "__main__":
    import wandb

    input_path = os.path.join(PROJECT_ROOT, 'Data/processed_tensor_data.npz')
    dataset = ChessDataset(input_path)
    wandb.init(project="chess-ai")

    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model = ResNet(num_residual_blocks=6).to(device)

    config = wandb.config
    config.learning_rate = 0.0001
    config.batch_size = 256
    config.epochs = 100

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    early_stopping = EarlyStopping(patience=5, delta=0.0001, path=os.path.join(PROJECT_ROOT, 'models/cnn.pth'))

    for epoch in range(config.epochs):
        train_loop(train_loader, model, loss_fn, optimizer, epoch)
        val_loss = test_loop(val_loader, model, loss_fn)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    print("Model trained!")

    wandb.finish()