# NebulaChess

NebulaChess is a UCI chess engine built in python that uses a residual neural network as an evaluation function. The network was trained using stockfish evaluations of around 20 million positions. The dataset is too large to be uploaded but can be found [here](https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations).

# Engine features:
- Residual neural network evaluation function
- Negamax search algorithm with alpha-beta pruning
- Iterative deepening
- Transposition table with zobrist hashing
- Move ordering
- Repetition table

# Installation

```
# Clone the repo
git clone https://github.com/benharvey1/NebulaChess.git

# Navigate to the directory
cd NebulaChess 

# Create a virtual environment
python -m venv venv

# Activate the virtual environment - Windows
.\venv\Scripts\activate

# Activate the virtual environment - Unix
source venv/bin/activate

# Install the requirements
pip install -r requirements.txt
```

# Usage

You can play the engine in your terminal with the command `python src/play_in_terminal.py`. This starts an interactive session where you can input moves in UCI notation (e.g. `e2e4`). You can also play the engine with your favourite chess GUI. To do this, simply upload the `NebulaChess.bat` file as an engine and set the protocol to "UCI".

# Improvements

There is definitely plenty of room for improvement. There are lots of classic chess engine tricks/optimisations that can be added to improve the search algorithm and the current evaluation function could be improved by training the neural net on a larger dataset. I think to see big improvements a slightly different approach would need to be taken - probably using a combination of classical evaluation and a NNUE like Stockfish does. Also, the code would need to be rewritten in a language like C++ (something I would like to do some day).

# Acknowledgements

This project utilised the [python-chess](https://github.com/niklasf/python-chess/) package. 

Inspirations:
- https://www.chessprogramming.org
- http://www.brucemo.com/compchess/programming/index.htm
- https://github.com/thomasahle/sunfish
- https://github.com/erikbern/deep-pink
- https://github.com/SebLague/Chess-Coding-Adventure
- https://github.com/geohot/twitchchess
- https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf
- https://arxiv.org/pdf/2007.02130




