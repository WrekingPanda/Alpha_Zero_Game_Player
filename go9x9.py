from CNN import Net
from alphazero import AlphaZero
from alphazeroparallel import AlphaZeroParallel
from az_parallel2 import AlphaZeroParallel2
from go import GoBoard
from torch.optim import Adam

def train():
    model = Net(9,(9**2)+1,40,64)
    board = GoBoard(9)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    params = {"n_iterations":10, "self_play_iterations":20, "mcts_iterations":100, "n_epochs":100, "batch_size":64, "n_self_play_parallel":5}
    Alpha = AlphaZeroParallel2(model, optimizer, board, 'G', data_augmentation=True, **params)
    Alpha.Learn()

train()