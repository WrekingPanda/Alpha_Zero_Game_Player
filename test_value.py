from ataxx import AttaxxBoard
import torch
torch.manual_seed(0)
from CNN import Net
from alpha_MCTS import MCTS
import numpy as np

import warnings
warnings.filterwarnings("ignore")

board = AttaxxBoard(4)
board.Start()
model = Net(4,4**4,10,128)
model.load_state_dict(torch.load("A4_hoje.pt", map_location="cuda" if torch.cuda.is_available() else "cpu"))
board.Move([0,0,0,2])
board.Move([3,3,3,1])
board.NextPlayer()
board.Fill()
print(board.board)
game_state = board.EncodedGameStateChanged()
game_state = torch.tensor(game_state, device=model.device).unsqueeze(0)
print(game_state)
policy,value = model(game_state)
#print(policy)
print(value)
board.NextPlayer()
game_state = board.EncodedGameStateChanged()
game_state = torch.tensor(game_state, device=model.device).unsqueeze(0)
print(game_state)
policy,value = model(game_state)
#print(policy)
print(value)
