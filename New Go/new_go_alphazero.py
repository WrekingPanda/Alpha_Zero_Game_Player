import numpy as np
from new_go_mcts import MCTSParallel
from no_oop_go import GoFunctional as GF, EncodedGameStateChanged, rotate90, flip_vertical

import random
from tqdm import tqdm
#from tqdm.notebook import tqdm

import torch
import torch.nn.functional as F
torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")

# for the data augmentation process
def transformations(board_array, player, action_probs, outcome):
    side = board_array.shape[0]
    transf = []
    transf.append((EncodedGameStateChanged(flip_vertical(board_array), player), np.append(np.flip(np.copy(action_probs)[:-1].reshape(side,side),0).flatten(),action_probs[-1]), outcome))                           # flip vertically
    transf.append((EncodedGameStateChanged(rotate90(board_array, 1), player), np.append(np.rot90(np.copy(action_probs)[:-1].reshape(side,side),1).flatten(),action_probs[-1]), outcome))                            # rotate 90
    transf.append((EncodedGameStateChanged(flip_vertical(rotate90(board_array, 1)), player), np.append(np.rot90(np.flip(np.copy(action_probs)[:-1].reshape(side,side),1),0).flatten(),action_probs[-1]), outcome))  # rotate 90 and flip vertically
    transf.append((EncodedGameStateChanged(rotate90(board_array, 2), player), np.append(np.rot90(np.copy(action_probs)[:-1].reshape(side,side),2).flatten(),action_probs[-1]), outcome))                            # rotate 180
    transf.append((EncodedGameStateChanged(flip_vertical(rotate90(board_array, 2)), player), np.append(np.rot90(np.flip(np.copy(action_probs)[:-1].reshape(side,side),1),0).flatten(),action_probs[-1]), outcome))  # rotate 180 and flip vertically
    transf.append((EncodedGameStateChanged(rotate90(board_array, 3), player), np.append(np.rot90(np.copy(action_probs)[:-1].reshape(side,side),3).flatten(),action_probs[-1]), outcome))                            # rotate 270
    transf.append((EncodedGameStateChanged(flip_vertical(rotate90(board_array, 3)), player), np.append(np.rot90(np.flip(np.copy(action_probs)[:-1].reshape(side,side),1),0).flatten(),action_probs[-1]), outcome))  # rotate 270 and flip vertically
    return transf

class AlphaZeroParallel2:
    # params = {n_iterations=10, self_play_iterations=10, mcts_iterations=100, n_epochs=10}
    def __init__(self, model, optimizer, board_size, gameType, data_augmentation=False, verbose=False, **params):
        self.model = model
        self.optimizer = optimizer
        self.board_size = board_size
        self.gameType = gameType
        self.params = params
        self.data_augmentation = data_augmentation
        self.verbose = verbose

    def SelfPlay(self):
        return_dataset = []
        boards = [None for _ in range(self.params["n_self_play_parallel"])]
        boards_dataset = [[] for _ in range(self.params["n_self_play_parallel"])]
        for i in range(self.params["n_self_play_parallel"]):
            boards[i] = (
                # board, cur_player, last two boards, winner
                np.zeros(shape=(self.board_size, self.board_size)), 1, [np.zeros(shape=(self.board_size, self.board_size)).tolist()]*3, 0
            )

        self.mcts = MCTSParallel(self.params["mcts_iterations"], self.model)

        while len(boards) > 0:
            boards_actions_probs = self.mcts.Search(boards)

            for i in range(len(boards))[::-1]:
                board_array, player, lt_boards, winner = boards[i]
                action_probs = boards_actions_probs[i]
                boards_dataset[i].append((np.copy(board_array), action_probs, player))
                moves = list(range(len(action_probs)))
                action = np.random.choice(moves, p=action_probs)
                move = self.mcts.roots[i].children[action].originMove
                next_board_array = np.copy(board_array)
                next_lt_boards = np.copy(lt_boards)
                GF.Move(next_board_array, move, player, next_lt_boards)
                next_lt_boards = next_lt_boards.tolist()
                next_player = GF.NextPlayer(player)
                next_winner = GF.CheckFinish(next_board_array, next_player, next_lt_boards)
                # update boards[i]
                boards[i] = (next_board_array, next_player, next_lt_boards, next_winner)

                #print("\nNEW BOARD Move:", move)
                #print(next_board_array)

                if next_winner != 0:
                    print("\nGAME ENDED\n")
                    boards_dataset[i].append((np.copy(next_board_array), action_probs, next_player)) # add the final config
                    for board_array, action_probs, player in boards_dataset[i]:
                        outcome = 1 if player==next_winner else -1
                        return_dataset.append((EncodedGameStateChanged(board_array, player), action_probs, outcome))
                        # data augmentation process (rotating and flipping the board)
                        if self.data_augmentation:
                            for transformed_data in transformations(board_array, player, action_probs, outcome):
                                return_dataset.append(transformed_data)
                    del boards[i]
        return return_dataset

    
    def Train(self, dataset):
        random.shuffle(dataset)
        for batch_index in range(0, len(dataset), self.params['batch_size']):
            sample = dataset[batch_index : batch_index+self.params["batch_size"]]
            board_encoded, policy_targets, value_targets = zip(*sample)
            board_encoded, policy_targets, value_targets = np.array(board_encoded), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            board_encoded = torch.tensor(board_encoded, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

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
            for sp_iteration in tqdm(range(self.params["self_play_iterations"]//self.params["n_self_play_parallel"]), desc="Self-Play Iterations", leave=False, unit="iter", ncols=100, colour="#fca965"):
                dataset += self.SelfPlay()
            
            self.model.train()
            for epoch in tqdm(range(self.params["n_epochs"]), desc="Training Model", leave=False, unit="epoch", ncols=100, colour="#9ffc65"):
                self.Train(dataset)
            
            torch.save(self.model.state_dict(), f"../Models/{str.upper(self.gameType)}{self.board.size}/{str.upper(self.gameType)}{self.board.size}_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"../Optimizers/{str.upper(self.gameType)}{self.board.size}/{str.upper(self.gameType)}{self.board.size}_{iteration}_opt.pt")
