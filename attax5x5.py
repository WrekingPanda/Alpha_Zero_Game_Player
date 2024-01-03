from CNN import Net
from az_parallel2 import AlphaZeroParallel2
from ataxx import AttaxxBoard
from torch.optim import Adam

def train():
    model = Net(5,5**4,10,128)
    board = AttaxxBoard(5)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    params = {"n_iterations":10, "self_play_iterations":100, "mcts_iterations":200, "n_epochs":20, "batch_size":128, "n_self_play_parallel":20}
    Alpha = AlphaZeroParallel2(model, optimizer, board, 'A', data_augmentation=False, **params)
    Alpha.Learn()

train()