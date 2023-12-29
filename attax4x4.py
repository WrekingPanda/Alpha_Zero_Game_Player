from CNN import Net
from alphazero import AlphaZero
from ataxx import AttaxxBoard
from torch.optim import Adam
import torch

def train():
    model = Net(4,4**4,40,64)
    board = AttaxxBoard(4)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    params = {"n_iterations":10, "self_play_iterations":20, "mcts_iterations":50, "n_epochs":50, "batch_size":64}
    Alpha = AlphaZero(model, optimizer, board, 'A', verbose=True, **params)
    Alpha.Learn()

train()