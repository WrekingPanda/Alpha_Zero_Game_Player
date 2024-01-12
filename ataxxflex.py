from CNN import Net
from az_parallel2 import AlphaZeroParallel2
from ataxx import AtaxxBoard
from torch.optim import Adam
FILL_SIZE = 10
A4_MODEL_PARAMS = {"size":FILL_SIZE, "action_size":FILL_SIZE**4, "num_resBlocks":10, "num_hidden":32} 
A4_TRAIN_PARAMS = {"n_iterations":2, "self_play_iterations":14, "mcts_iterations":5, "n_epochs":2, "batch_size":128, "n_self_play_parallel":2, "move_cap":20}

def train():
    model = Net(**A4_MODEL_PARAMS)
    board = AtaxxBoard(4)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    Alpha = AlphaZeroParallel2(model, optimizer, board, 'A', data_augmentation=True, **A4_TRAIN_PARAMS, fill_size=FILL_SIZE)
    Alpha.Learn()

train()