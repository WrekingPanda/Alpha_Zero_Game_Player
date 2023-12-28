from CNN import Net
from alphazero import AlphaZero
from ataxx import AttaxxBoard
from torch.optim import Adam
import torch

def train():
    model = Net(4,4**4,40,64)
    board = AttaxxBoard(4)
    optimizer = Adam(model.parameters(), lr=0.07)
    params = {"n_iterations":10, "self_play_iterations":20, "mcts_iterations":400, "n_epochs":100, "batch_size":64}
    Alpha = AlphaZero(model, optimizer, board, 'A', **params)
    Alpha.Learn()

train()