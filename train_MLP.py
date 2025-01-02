import numpy as np
from tqdm import tqdm
from chess_dataset import ChessDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split

# NOTE : Training not going to well atm
# Things to try...
    # TODO: try training with cutoff at +- 1000 eval
    # TODO: try training without the 'possible moves' in feature rep 
    # TODO: try training with more data (smooth out distribution)
    # TODO: Might have to try different feature rep (piece co-ordinates)
    # TODO: try different transformations (tanh, arctan etc.)


class MLP(nn.Module):
    """Artificial Neural Network"""

    def __init__(self):

        super(MLP, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(901, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.output = nn.Linear(2048, 1)

    def forward(self, x):
        """Input vector of size 901.
        3 hidden layers with 2048 units. ReLU activation and dropout 
        with probability 0.25 between each layer. Tanh activation for output layer. 
        """
        x = self.dropout(F.relu(self.fc1(x)))   # input: (batch_size,901), output: (batch_size,2048)
        x = self.dropout(F.relu(self.fc2(x)))   # input: (batch_size,2048), output: (batch_size,2048)
        x = self.dropout(F.relu(self.fc3(x)))   # input: (batch_size,2048), output: (batch_size,2048)
        x = torch.tanh(self.output(x))  # input: (batch_size,2048), output: (batch_size, 1)

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
            loss = loss_fn(pred, y.unsqueeze(-1)) # need to resahpe y from (256,1) to (256,)

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

    input_path = 'processed_vector_data.npz'
    dataset = ChessDataset(input_path)
    wandb.init(project="chess-ai", name=f"lr=0.0001_bs=256_dr=0.3")

    train_val_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.05, random_state=42)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.2, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    test_X, test_y = [], []
    for batch in test_loader:
        inputs, labels = batch
        test_X.append(inputs.numpy())
        test_y.append(labels.numpy())

    test_X = np.concatenate(test_X, axis=0)
    test_y = np.concatenate(test_y, axis=0)
    np.savez_compressed("test_vector_data.npz", X=test_X, y=test_y)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model = MLP().to(device)
    #model.load_state_dict(torch.load("NeuralNet_2.pth"))

    config = wandb.config
    config.learning_rate = 0.0001
    config.batch_size = 256
    config.epochs = 100

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    early_stopping = EarlyStopping(patience=5, delta=0.0001, path='MLP.pth')

    for epoch in range(config.epochs):
        train_loop(train_loader, model, loss_fn, optimizer, epoch)
        val_loss = test_loop(val_loader, model, loss_fn)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    print("Model trained!")

    wandb.finish()






