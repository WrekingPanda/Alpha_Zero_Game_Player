import torch
from alpha_MCTS import MCTS

class AlphaZero:
    # params = {n_iterations=10, self_play_iterations=10, mcts_iterations=100, n_epochs=10}
    def __init__(self, model, optimizer, board, **params):
        self.model = model
        self.optimizer = optimizer
        self.board = board
        self.params = params
        self.mcts = MCTS()

    def SelfPlay(self):
        pass
    
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

        