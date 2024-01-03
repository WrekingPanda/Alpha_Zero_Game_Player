from CNN import Net
from az_parallel2 import AlphaZeroParallel2
from go import GoBoard
from torch.optim import Adam

def train():
    model = Net(7,(7**2)+1,20,256)
    board = GoBoard(7)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    params = {"n_iterations":10, "self_play_iterations":20, "mcts_iterations":100, "n_epochs":20, "batch_size":128, "n_self_play_parallel":10}
    Alpha = AlphaZeroParallel2(model, optimizer, board, 'G', data_augmentation=False, **params)
    Alpha.Learn()

train()