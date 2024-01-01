import numpy as np 
import torch
from no_oop_go import GoFunctional as GF, EncodedGameStateChanged

from tqdm import tqdm
#from tqdm.notebook import tqdm

class MCTS_Node:
    def __init__(self, board, player, last_two_boards, winner, parent=None, move=None, policy_value=0) -> None:
        self.board: np.ndarray = board
        self.player = player
        self.last_two_boards = last_two_boards
        self.winner = winner
        self.w = 0 # Sum of backpropagation 
        self.n = 0 # Num of visits
        self.p = policy_value # Probability returned from NN
        self.originMove = move
        self.parent = parent
        self.children = {} # Save all children 

    def Select(self):
        c = 2
        max_ucb = -np.inf
        best_node = []
        if len(self.children) == 0:
            return self
        for child in self.children.values():
            # Calculate UCB for each children
            ucb = child.w/child.n + child.p*c*(self.n**(1/2))/(1+child.n) if child.n != 0 else child.p*c*(self.n**(1/2))/(1+child.n)
            # Update max UCB value, as well as best Node
            if ucb > max_ucb: 
                max_ucb = ucb
                best_node = [child]
            elif ucb == max_ucb:
                best_node.append(child)
        return np.random.choice(best_node)

    def Expansion(self, policy):
        if len(self.children) != 0 or self.winner != 0:
            return
        possibleMoves = GF.PossibleMoves(self.board, self.player, self.last_two_boards)
        for move in possibleMoves:
            child_board = np.copy(self.board)
            last_two_boards_copy = np.copy(self.last_two_boards)
            GF.Move(child_board, move, self.player, last_two_boards_copy)
            action = GF.MoveToAction(move, child_board.shape[0])
            player = GF.NextPlayer(self.player)
            last_two_boards_copy = last_two_boards_copy.tolist()
            winner = GF.CheckFinish(child_board, player, last_two_boards_copy) if len(last_two_boards_copy) >= 3 else 0
            self.children[action] = MCTS_Node(child_board, player, last_two_boards_copy, winner, self, move, policy[action])

    def BackPropagation(self, value):
        self.w += value
        self.n += 1
        if self.parent is not None:
            self.parent.BackPropagation(-1*value)


class MCTSParallel:
    def __init__(self, n_iterations, model, dirichlet_eps=0.25) -> None:
        self.n_iterations = n_iterations
        self.model = model
        self.dirichlet_eps = dirichlet_eps
        self.roots = []

    @torch.no_grad()
    def Search(self, board_arrays_list):
        self.roots = [MCTS_Node(board, player, lt_boards, winner) for board, player, lt_boards, winner in board_arrays_list]

        # add noise to the roots' policy array
        boards_states = [EncodedGameStateChanged(root.board, root.player) for root in self.roots]
        boards_states = torch.tensor(boards_states, device=self.model.device)
        policy, _ = self.model(boards_states)
        policy = policy.cpu().numpy()
        action_space_size = (self.roots[0].board.shape[0]**2)+1
        dirichlet_arr = np.ones(shape=action_space_size)/action_space_size
        policy = (1-self.dirichlet_eps)*policy + self.dirichlet_eps*np.random.dirichlet(dirichlet_arr)
        # expand each root and associate the policy valyes with the ramifications
        for i in range(len(board_arrays_list)):
            self.roots[i].Expansion(policy[i])
            self.roots[i].n = 1

        # start MCTS iterations
        for _ in tqdm(range(self.n_iterations), desc="MCTS Iterations", leave=False, unit="iter", ncols=100, colour="#f7fc65"):             
            nodes = [self.roots[i].Select() for i in range(len(board_arrays_list))]
            # selection phase
            for i in range(len(board_arrays_list)):
                while len(nodes[i].children) > 0:
                    nodes[i] = nodes[i].Select()            
            # get the values from NN
            boards_states = [EncodedGameStateChanged(nodes[i].board, nodes[i].player) for i in range(len(board_arrays_list))]
            boards_states = torch.tensor(boards_states, device=self.model.device)
            policy, value = self.model(boards_states)
            policy = policy.cpu().numpy()
            value = value.cpu().numpy()
            for i in range(len(board_arrays_list)):
                # expansion phase
                nodes[i].Expansion(policy[i])
                # backprop phase
                nodes[i].BackPropagation(value[i][0])
        
        # return the actions' probabilities for each game
        action_space_size = (self.roots[0].board.shape[0]**2)+1
        boards_actions_probs = [np.zeros(shape=action_space_size) for _ in range(len(board_arrays_list))]
        for i in range(len(board_arrays_list)):
            for action, child in self.roots[i].children.items():
                boards_actions_probs[i][action] = child.n
            boards_actions_probs[i] /= np.sum(boards_actions_probs[i])
        return boards_actions_probs

        # to obtain the resulting move
        # final_action = action_probs.argmax()
        # root.children[final_action].originMove
        