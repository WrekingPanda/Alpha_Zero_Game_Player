from CNN import Net
from az_parallel2 import AlphaZeroParallel2
from ataxx import AttaxxBoard
from torch.optim import Adam

A4_MODEL_PARAMS = {"size":4, "action_size":4**4, "num_resBlocks":10, "num_hidden":32} 
A4_TRAIN_PARAMS = {"n_iterations":10, "self_play_iterations":20, "mcts_iterations":150, "n_epochs":20, "batch_size":128, "n_self_play_parallel":10, "move_cap":4}

def train():
    model = Net(**A4_MODEL_PARAMS)
    board = AttaxxBoard(4)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    Alpha = AlphaZeroParallel2(model, optimizer, board, 'A', data_augmentation=True, **A4_TRAIN_PARAMS)
    Alpha.Learn()

train()