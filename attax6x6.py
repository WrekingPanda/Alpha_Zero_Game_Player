from CNN import Net
from alphazero import AlphaZero
from alphazeroparallel import AlphaZeroParallel
from az_parallel2 import AlphaZeroParallel2
from ataxx import AttaxxBoard
from torch.optim import Adam

def train():
    model = Net(6,6**4,40,64)
    board = AttaxxBoard(6)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    params = {"n_iterations":10, "self_play_iterations":3, "mcts_iterations":100, "n_epochs":50, "batch_size":64, "n_self_play_parallel":2}
    Alpha = AlphaZeroParallel2(model, optimizer, board, 'A', data_augmentation=True, **params)
    Alpha.Learn()

train()