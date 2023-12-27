import torch
import numpy as np
from alpha_MCTS import MCTS
from ataxx import AttaxxBoard
from go import GoBoard

import random

import torch.nn.functional as F
torch.manual_seed(0)

class AlphaZero:
    # params = {n_iterations=10, self_play_iterations=10, mcts_iterations=100, n_epochs=10}
    def __init__(self, model, optimizer, board, gameType, **params):
        self.model = model
        self.optimizer = optimizer
        self.board = board
        self.gameType = gameType
        self.params = params

    def SelfPlay(self):
        dataset = []
        board = AttaxxBoard(self.board.size) if self.gameType == "A" else GoBoard(self.board.size)
        board.Start(render=False)
        self.mcts = MCTS(board, self.params["mcts_iterations"], self.model)

        while True:
            action_probs = self.mcts.Search()

            dataset.append((board, action_probs, board.player)) 
            action_list = list(range(len(action_probs)))
            action = np.ramdom.choice(action_list, p=action_probs)
            move = self.mcts.root.children[action].originMove
            board.Move(move)

            if board.hasFinished():
                return_dataset = []
                for board, action_probs, player in dataset:
                    outcome = 1 if player==board.winner else -1
                    return_dataset.append((board.EncodedGameState(), action_probs, outcome))
                return return_dataset
            
            board.NextPlayer()

    
    def Train(self, dataset):
        random.shuffle(dataset)
        for batch_index in range(0, len(dataset), self.params['batch_size']):
            sample = dataset[batch_index:min(len(dataset)-1, batch_index + self.params["batch_size"])]
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
        for iteration in range(self.params["n_iterations"]):
            dataset = []

            self.model.eval()
            for sp_iteration in range(self.params["self_play_iterations"]):
                dataset += self.SelfPlay()
            
            self.model.train()
            for epoch in range(self.params["n_epochs"]):
                self.Train(dataset)
            
            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")


        