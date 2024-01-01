from ataxx import AttaxxBoard
import torch
from CNN import Net
from alpha_MCTS import MCTS
import numpy as np

import warnings
warnings.filterwarnings("ignore")

board = AttaxxBoard(4)
board.Start()
model = Net(4,4**4,40,64)
model.load_state_dict(torch.load("A4.pt", map_location="cuda" if torch.cuda.is_available() else "cpu"))
model.eval()
mcts = MCTS(board, 75, model)

while True:
    print(board.board)

    if board.player == 1:
        i1, j1, i2, j2 = list(map(int, input("Move: ").split(" ")))
        if board.ValidMove(i1, j1, i2, j2):
            board.Move((i1, j1, i2, j2))
            board.NextPlayer()
            board.CheckFinish()
        else:
            print("NOT A VALID MOVE")

    else:
        mcts_probs = mcts.Search(board)
        action = np.argmax(mcts_probs)
        move = mcts.root.children[action].originMove
        print("Alphazero Move:", move)
        board.Move(move)
        board.NextPlayer()
        board.CheckFinish()
    
    if board.hasFinished():
        print(board.board)
        if board.winner != 3:
            print(board.winner, "won")
        else:
            print("draw")
        break