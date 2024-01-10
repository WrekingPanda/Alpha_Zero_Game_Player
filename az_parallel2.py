import numpy as np
from aMCTS_parallel import MCTSParallel, MCTS_Node
from ataxx import AttaxxBoard
from go import GoBoard

import random
#from tqdm import tqdm
from tqdm.notebook import tqdm

import torch
import torch.nn.functional as F
torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")

# for the data augmentation process
def transformations(board_state, action_probs, outcome, gameType):
    if gameType == 'G':
        side = board_state.size
        transf = []
        transf.append((board_state.flip_vertical().EncodedGameStateChanged(), np.append(np.flip(np.copy(action_probs)[:-1].reshape(side,side),0).flatten(),action_probs[-1]), outcome))                         # flip vertically
        transf.append((board_state.rotate90(1).EncodedGameStateChanged(), np.append(np.rot90(np.copy(action_probs)[:-1].reshape(side,side),1).flatten(),action_probs[-1]), outcome))                            # rotate 90
        transf.append((board_state.rotate90(1).flip_vertical().EncodedGameStateChanged(), np.append(np.rot90(np.flip(np.copy(action_probs)[:-1].reshape(side,side),1),0).flatten(),action_probs[-1]), outcome)) # rotate 90 and flip vertically
        transf.append((board_state.rotate90(2).EncodedGameStateChanged(), np.append(np.rot90(np.copy(action_probs)[:-1].reshape(side,side),2).flatten(),action_probs[-1]), outcome))                            # rotate 180
        transf.append((board_state.rotate90(2).flip_vertical().EncodedGameStateChanged(), np.append(np.rot90(np.flip(np.copy(action_probs)[:-1].reshape(side,side),1),0).flatten(),action_probs[-1]), outcome)) # rotate 180 and flip vertically
        transf.append((board_state.rotate90(3).EncodedGameStateChanged(), np.append(np.rot90(np.copy(action_probs)[:-1].reshape(side,side),3).flatten(),action_probs[-1]), outcome))                            # rotate 270
        transf.append((board_state.rotate90(3).flip_vertical().EncodedGameStateChanged(), np.append(np.rot90(np.flip(np.copy(action_probs)[:-1].reshape(side,side),1),0).flatten(),action_probs[-1]), outcome)) # rotate 270 and flip vertically
        return transf
    elif gameType == 'A':
        side = board_state.size
        transf = []
        transf.append((board_state.flip_vertical().EncodedGameStateChanged(), np.flip(np.flip(np.copy(action_probs).reshape(side,side,side,side),2),0).flatten(), outcome))                                                 # flip vertically
        transf.append((board_state.rotate90(1).EncodedGameStateChanged(), np.rot90(np.rot90(np.copy(action_probs).reshape(side,side,side,side),1,(2,3)),1,(0,1)).flatten(), outcome))                                       # rotate 90
        transf.append((board_state.rotate90(1).flip_vertical().EncodedGameStateChanged(), np.flip(np.flip(np.rot90(np.rot90(np.copy(action_probs).reshape(side,side,side,side),1,(2,3)),1,(0,1)),2),0).flatten(), outcome)) # rotate 90 and flip vertically
        transf.append((board_state.rotate90(2).EncodedGameStateChanged(), np.rot90(np.rot90(np.copy(action_probs).reshape(side,side,side,side),2,(2,3)),2,(0,1)).flatten(), outcome))                                       # rotate 180
        transf.append((board_state.rotate90(2).flip_vertical().EncodedGameStateChanged(), np.flip(np.flip(np.rot90(np.rot90(np.copy(action_probs).reshape(side,side,side,side),2,(2,3)),2,(0,1)),2),0).flatten(), outcome)) # rotate 180 and flip vertically
        transf.append((board_state.rotate90(3).EncodedGameStateChanged(), np.rot90(np.rot90(np.copy(action_probs).reshape(side,side,side,side),3,(2,3)),3,(0,1)).flatten(), outcome))                                       # rotate 270
        transf.append((board_state.rotate90(3).flip_vertical().EncodedGameStateChanged(), np.flip(np.flip(np.rot90(np.rot90(np.copy(action_probs).reshape(side,side,side,side),3,(2,3)),3,(0,1)),2),0).flatten(), outcome)) # rotate 270 and flip vertically
        return transf
    return []


# function that applies temperature to the given probabilities distribution and normalizes the result, for the current AlphaZero iteration
def probs_with_temperature(probabilities, az_iteration):
    # returns a vale between 1.25 and 0.75
    def temperature_function(az_iter):
        return 1 / (1 + np.e**(az_iter-5)) + 0.5
    prob_temp =  probabilities**(1/temperature_function(az_iteration))
    prob_temp /= np.sum(prob_temp)
    return prob_temp


class AlphaZeroParallel2:
    # params = {n_iterations=10, self_play_iterations=10, mcts_iterations=100, n_epochs=10}
    def __init__(self, model, optimizer, board, gameType, data_augmentation=False, verbose=False, **params):
        self.model = model
        self.optimizer = optimizer
        self.board = board
        self.gameType = gameType
        self.params = params
        self.data_augmentation = data_augmentation
        self.verbose = verbose

    def SelfPlay(self, az_iteration):
        return_dataset = []
        selfplays_done = 0
        boards = [None for _ in range(self.params["n_self_play_parallel"])]
        boards_dataset = [[] for _ in range(self.params["n_self_play_parallel"])]
        for i in range(self.params["n_self_play_parallel"]):
            boards[i] = AttaxxBoard(self.board.size) if self.gameType == "A" else GoBoard(self.board.size)
            boards[i].Start(render=False)

        self.mcts = MCTSParallel(self.model)
        root_boards = [MCTS_Node(board) for board in boards]
        while len(boards) > 0:

            # print("\nINSIDE SELFPLAY")
            # for root_board in root_boards:
            #     print(root_board.originMove, root_board.n)
            # print()
            
            boards_actions_probs = self.mcts.Search(root_boards, self.params["mcts_iterations"])
            for i in range(len(boards))[::-1]:
                action_probs = boards_actions_probs[i]
                #print("\nACTION PROBS GAME", i)
                #print(action_probs)
                #print()
                # apply temperature
                #action_probs = probs_with_temperature(action_probs, az_iteration)
                boards_dataset[i].append((boards[i].copy(), action_probs, boards[i].player))
                moves = list(range(len(action_probs)))
                action = np.random.choice(moves, p=action_probs)
                move = self.mcts.roots[i].children[action].originMove
                boards[i].Move(move)
                boards[i].NextPlayer()
                boards[i].CheckFinish()

                # if i == 0:
                #     print("\nFIRST BOARD GAME")
                #     print(boards[0].board)

                # update the new root (root is now the played child state)
                root_boards[i] = self.mcts.roots[i].children[action]
                root_boards[i].parent = None # it is needed to "remove" / "delete" the parent state

                if boards[i].hasFinished():
                    boards_dataset[i].append((boards[i].copy(), action_probs, boards[i].player)) # add the final config
                    boards[i].NextPlayer()
                    boards_dataset[i].append((boards[i].copy(), action_probs, boards[i].player))

                    for board, action_probs, player in boards_dataset[i]:
                        outcome = 1 if player==boards[i].winner else -1
                        return_dataset.append((board.EncodedGameStateChanged(), action_probs, outcome))
                        # data augmentation process (rotating and flipping the board)
                        if self.data_augmentation:
                            for transformed_data in transformations(board, action_probs, outcome, self.gameType):
                                return_dataset.append(transformed_data)
                    # dynamic parallel selfplay allocation
                    selfplays_done += 1
                    if selfplays_done % self.params["n_self_play_parallel"] == 0:
                        print("\nSELFPLAY:", selfplays_done * 100 // self.params["self_play_iterations"], "%")
                    if selfplays_done >= self.params["self_play_iterations"] - self.params["n_self_play_parallel"]:
                        del boards[i]
                        del root_boards[i]
                    else:
                        boards[i] = AttaxxBoard(self.board.size) if self.gameType == "A" else GoBoard(self.board.size)
                        boards[i].Start(render=False)
                        root_boards[i] = MCTS_Node(boards[i])
                        boards_dataset[i] = []

        print("\nSELFPLAY: 100 %")
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
        for az_iteration in tqdm(range(self.params["n_iterations"]), desc="AlphaZero Algorithm Iterations", leave=False, unit="iter", ncols=100, colour="#fc6a65"):
            self.model.eval()
            dataset = self.SelfPlay(az_iteration)
            
            self.model.train()
            for epoch in tqdm(range(self.params["n_epochs"]), desc="Training Model", leave=False, unit="epoch", ncols=100, colour="#9ffc65"):
                self.Train(dataset)
            
            torch.save(self.model.state_dict(), f"./Models/{str.upper(self.gameType)}{self.board.size}/{str.upper(self.gameType)}{self.board.size}_{az_iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"./Optimizers/{str.upper(self.gameType)}{self.board.size}/{str.upper(self.gameType)}{self.board.size}_{az_iteration}_opt.pt")
