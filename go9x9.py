from CNN import Net
from az_parallel2 import AlphaZeroParallel2
from go import GoBoard
from torch.optim import Adam

G9_MODEL_PARAMS = {"size":9, "action_size":9**2+1, "num_resBlocks":30, "num_hidden":64} 
G9_TRAIN_PARAMS = {"n_iterations":10, "self_play_iterations":40, "mcts_iterations":200, "n_epochs":20, "batch_size":128, "n_self_play_parallel":20, "move_cap":round(9**(2.5))}

def train():
    model = Net(**G9_MODEL_PARAMS)
    board = GoBoard(9)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    Alpha = AlphaZeroParallel2(model, optimizer, board, 'G', data_augmentation=False, **G9_TRAIN_PARAMS)
    Alpha.Learn()

train()