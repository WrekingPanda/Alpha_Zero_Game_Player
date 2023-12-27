import torch
import numpy as np
from alpha_MCTS import MCTS
from ataxx import AttaxxBoard
from go import GoBoard


class AlphaZero:
    # params = {n_iterations=10, self_play_iterations=10, mcts_iterations=100, n_epochs=10}
    def __init__(self, model, optimizer, board, gameType, **params):
        self.model = model
        self.optimizer = optimizer
        self.board = board
        self.gameType = gameType
        self.params = params
        self.mcts = MCTS()


    def SelfPlay(self):
        dataset = []

        if self.gameType == 'A':
            board = AttaxxBoard()
        elif self.gameType == 'G':
            board = GoBoard()

        board.Start(render=False)
        self.mcts.board = board

        while True:
            action_probs = self.mcts.Search()

            dataset.append((board.board, action_probs, board.player)) 
            action_list = list(range(len(action_probs)))
            action = np.ramdom.choice(action_list, p=action_probs)

            move = self.mcts.root.children[action].originMove
            board.Move(move)

            



    
    def Train(self, dataset):
        pass


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


        