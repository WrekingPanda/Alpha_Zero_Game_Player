from CNN import Net
#from alphazero import AlphaZero
#from alphazeroparallel import AlphaZeroParallel
from new_go_alphazero import AlphaZeroParallel2
#from go import GoBoard
from torch.optim import Adam

def train():
    model = Net(7,(7**2)+1,40,64)
    #board = GoBoard(9)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    params = {"n_iterations":10, "self_play_iterations":20, "mcts_iterations":100, "n_epochs":100, "batch_size":64, "n_self_play_parallel":1}
    Alpha = AlphaZeroParallel2(model, optimizer, 7, 'G', data_augmentation=True, **params)
    Alpha.Learn()

train()