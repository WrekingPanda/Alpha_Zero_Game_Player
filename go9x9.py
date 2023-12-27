from CNN import Net
from alphazero import AlphaZero
from go import GoBoard
from torch.optim import Adam
import torch

def train():
    model = Net(9,(9**2)+1,40,64)
    board = GoBoard(9)
    optimizer = Adam(model.parameters(), lr=0.07)
    params = {"n_iterations":2, "self_play_iterations":3, "mcts_iterations":10, "n_epochs":10, "batch_size":64}
    Alpha = AlphaZero(model, optimizer, board, 'G', **params)
    Alpha.Learn()

train()