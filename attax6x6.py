from CNN import Net
from alphazero import AlphaZero
from ataxx import AttaxxBoard
from torch.optim import Adam
import torch

def train():
    model = Net(6,6**4,40,64)
    board = AttaxxBoard(6)
    optimizer = Adam(model.parameters(), lr=0.07)
    params = {"n_iterations":2, "self_play_iterations":3, "mcts_iterations":10, "n_epochs":10, "batch_size":64}
    Alpha = AlphaZero(model, optimizer, board, 'A', **params)
    Alpha.Learn()

train()