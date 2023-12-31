from CNN import Net
from alphazero import AlphaZero
from ataxx import AttaxxBoard
from torch.optim import Adam
import torch

FILL_SIZE = 10

def train():
    model = Net(FILL_SIZE,FILL_SIZE**4,40,64)
    board = AttaxxBoard(FILL_SIZE)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    params = {"n_iterations":2, "self_play_iterations":1, "mcts_iterations":5, "n_epochs":5, "batch_size":64} #self_play_iterations deve ser multiplo de FILL_SIZE-3
    Alpha = AlphaZero(model, optimizer, board, 'A', verbose=True, fill_size=FILL_SIZE, **params)
    Alpha.Learn()

train()