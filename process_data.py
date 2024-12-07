import pandas as pd
import numpy as np
from Board_State import State
from tqdm import tqdm

data = pd.read_csv('Data.csv')
fen = data['FEN'].to_numpy()
evaluation = data['Evaluation'].to_numpy()

num_samples = len(data)
tensor_shape = (19, 8, 8)
vector_shape = (901,)

X_1 = np.zeros((num_samples, *tensor_shape), dtype=np.uint8)
X_2 = np.zeros((num_samples, *vector_shape), dtype=np.uint8)
y = np.zeros(num_samples, dtype=np.float32)

state_converter = State()

for i in tqdm(range(num_samples), desc='Processing Data', unit='Iteration'):
    state_converter.board.set_fen(fen[i])
    X_1[i] = state_converter.board_to_tensor()
    X_2[i] = state_converter.board_to_vector()
    y[i] = (2/np.pi)*np.arctan(evaluation[i]/250)   # arctan transformation 

np.savez_compressed('processed_tensor_data.npz', X=X_1, y=y)
np.savez_compressed('processed_vector_data.npz', X=X_2, y=y)
print("Data preprocessing complete!")




