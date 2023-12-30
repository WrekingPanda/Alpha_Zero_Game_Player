from CNN import Net
from alphazero import AlphaZero
from alphazeroparallel import AlphaZeroParallel
from go import GoBoard
from torch.optim import Adam

def train():
    model = Net(7,(7**2)+1,40,64)
    board = GoBoard(7)
    optimizer = Adam(model.parameters(), lr=0.07)
    params = {"n_iterations":2, "self_play_iterations":3, "mcts_iterations":10, "n_epochs":10, "batch_size":64, "n_self_play_parallel":2}
    Alpha = AlphaZeroParallel(model, optimizer, board, 'G', data_augmentation=True, **params)
    Alpha.Learn()

train()