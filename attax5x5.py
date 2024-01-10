from CNN import Net
from az_parallel2 import AlphaZeroParallel2
from ataxx import AttaxxBoard
from torch.optim import Adam

A5_MODEL_PARAMS = {"size":5, "action_size":5**4, "num_resBlocks":10, "num_hidden":64} 
A5_TRAIN_PARAMS = {"n_iterations":10, "self_play_iterations":100, "mcts_iterations":150, "n_epochs":20, "batch_size":128, "n_self_play_parallel":20}

def train():
    model = Net(**A5_MODEL_PARAMS)
    board = AttaxxBoard(5)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    Alpha = AlphaZeroParallel2(model, optimizer, board, 'A', data_augmentation=True, **A5_TRAIN_PARAMS)
    Alpha.Learn()

train()