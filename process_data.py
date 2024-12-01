import pandas as pd
import numpy as np
from Board_State import State
from tqdm import tqdm

data = pd.read_csv('TransformedData.csv')
fen = data['FEN'].to_numpy()
evaluation = data['Evaluation'].to_numpy()

num_samples = len(data)
tensor_shape = (5, 8, 8)

X = np.zeros((num_samples, *tensor_shape), dtype=np.uint8)
y = np.zeros(num_samples, dtype=np.float32)

state_converter = State()

for i in tqdm(range(num_samples), desc='Processing Data', unit='Iteration'):
    state_converter.board.set_fen(fen[i])
    X[i] = state_converter.board_to_tensor() 
    y[i] = float(evaluation[i]) 

np.savez_compressed('processed_data.npz', X=X, y=y)
print("Data preprocessing complete!")




