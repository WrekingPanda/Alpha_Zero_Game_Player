import numpy as np
from alpha_MCTS import MCTS
from ataxx import AttaxxBoard
from go import GoBoard

import random
from tqdm import tqdm
#from tqdm.notebook import tqdm

import torch
import torch.nn.functional as F
torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")

# for the data augmentation process
def transformations(board_state, action_probs, outcome):
    transf = []
    transf.append((board_state.flip_vertical().EncodedGameStateChanged(), action_probs, outcome))             # flip vertically
    transf.append((board_state.rotate90(1).EncodedGameStateChanged(), action_probs, outcome))                 # rotate 90
    transf.append((board_state.rotate90(1).flip_vertical().EncodedGameStateChanged(), action_probs, outcome)) # rotate 90 and flip vertically
    transf.append((board_state.rotate90(2).EncodedGameStateChanged(), action_probs, outcome))                 # rotate 180
    transf.append((board_state.rotate90(2).flip_vertical().EncodedGameStateChanged(), action_probs, outcome)) # rotate 180 and flip vertically
    transf.append((board_state.rotate90(3).EncodedGameStateChanged(), action_probs, outcome))                 # rotate 270
    transf.append((board_state.rotate90(3).flip_vertical().EncodedGameStateChanged(), action_probs, outcome)) # rotate 270 and flip vertically
    return transf

class AlphaZero:
    # params = {n_iterations=10, self_play_iterations=10, mcts_iterations=100, n_epochs=10}
    def __init__(self, model, optimizer, board, gameType, data_augmentation=False, **params):
        self.model = model
        self.optimizer = optimizer
        self.board = board
        self.gameType = gameType
        self.params = params
        self.data_augmentation = data_augmentation

    def SelfPlay(self, verbose=False):
        dataset = []
        board = AttaxxBoard(self.board.size) if self.gameType == "A" else GoBoard(self.board.size)
        board.Start(render=False)
        self.mcts = MCTS(board, self.params["mcts_iterations"], self.model)

        if verbose:
            with open(f"./selfplay_{self.gameType}{board.size}.txt", 'w') as file:
                file.write(f"")
                file.close()

        while True:
            action_probs = self.mcts.Search()

            dataset.append((board, action_probs, board.player)) 
            action_list = list(range(len(action_probs)))
            action = np.random.choice(action_list, p=action_probs)
            move = self.mcts.root.children[action].originMove
            board.Move(move)
            if verbose:
                with open(f"./selfplay_{self.gameType}{board.size}.txt", 'a') as file:
                    file.write(f"[Player {board.player}] Move: {move}\n{board.board}\n")
                    file.close()
            board.NextPlayer()
            board.CheckFinish()

            if board.hasFinished():
                if verbose:
                    with open(f"./selfplay_{self.gameType}{board.size}.txt", 'a') as file:
                        file.write(f"Winner: {board.winner}\n\n====================\n")
                        file.close()
                return_dataset = []
                for board, action_probs, player in dataset:
                    outcome = 1 if player==board.winner else -1
                    return_dataset.append((board.EncodedGameStateChanged(), action_probs, outcome))
                    # data augmentation process (rotating and flipping the board)
                    if self.data_augmentation:
                        for transformed_data in transformations(board, action_probs, outcome):
                            return_dataset.append(transformed_data)
                return return_dataset

    
    def Train(self, dataset):
        random.shuffle(dataset)
        for batch_index in range(0, len(dataset), self.params['batch_size']):
            sample = dataset[batch_index : batch_index+self.params["batch_size"]]
            board_encoded, policy_targets, value_targets = zip(*sample)
            board_encoded, policy_targets, value_targets = np.array(board_encoded), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            board_encoded = torch.tensor(board_encoded, dtype=torch.float32)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
            value_targets = torch.tensor(value_targets, dtype=torch.float32)

            out_policy, out_value = self.model(board_encoded)
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def Learn(self):
        for iteration in tqdm(range(self.params["n_iterations"]), desc="AlphaZero Algorithm Iterations", leave=False, unit="iter", ncols=100, colour="#fc6a65"):
            dataset = []

            self.model.eval()
            for sp_iteration in tqdm(range(self.params["self_play_iterations"]), desc="Self-Play Iterations", leave=False, unit="iter", ncols=100, colour="#fca965"):
                dataset += self.SelfPlay(verbose=True)
            
            self.model.train()
            for epoch in tqdm(range(self.params["n_epochs"]), desc="Training Model", leave=False, unit="epoch", ncols=100, colour="#9ffc65"):
                self.Train(dataset)
            
            torch.save(self.model.state_dict(), f"./Models/{str.upper(self.gameType)}{self.board.size}/{str.upper(self.gameType)}{self.board.size}_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"./Optimizers/{str.upper(self.gameType)}{self.board.size}/{str.upper(self.gameType)}{self.board.size}_{iteration}_opt.pt")


        