import numpy 
from ataxx import AttaxxBoard
from go import GoBoard
from copy import deepcopy
import torch

from tqdm import tqdm
#from tqdm.notebook import tqdm

class MCTS_Node:
    def __init__(self, board, parent=None, move=None, policy_value=0) -> None:
        self.board = board
        self.w = 0 # Sum of backpropagation 
        self.n = 0 # Num of visits
        self.p = policy_value # Probability returned from NN
        
        self.originMove = move
        self.parent = parent
        self.children = {} # Save all children 

    def Select(self):
        c = 2
        max_ucb = -numpy.inf
        best_node = []
        if len(self.children) == 0:
            return self
        for child in self.children.values():
            # Calculate UCB for each children
            ucb = child.w/child.n + child.p*c*(self.n**(1/2))/(1+child.n) if child.n != 0 else 0

            # Update max UCB value, as well as best Node
            if ucb > max_ucb: 
                max_ucb = ucb
                best_node = [child]
            elif ucb == max_ucb:
                best_node.append(child)
        return numpy.random.choice(best_node)

    def Expansion(self, policy):
        if len(self.children) != 0 or self.board.winner != 0:
            return
        possibleMoves = self.board.PossibleMoves()
        for move in possibleMoves:
            cur_board = deepcopy(self.board)
            cur_board.Move(move)
            action = cur_board.MoveToAction(move)
            cur_board.NextPlayer()
            cur_board.CheckFinish()
            self.children[action] = MCTS_Node(cur_board, self, move, policy_value=policy[action])

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
    def Search(self, boards_obj_list):
        self.roots = [MCTS_Node(board) for board in boards_obj_list]

        # add noise to the roots' policy array
        boards_states = [root.board.EncodedGameStateChanged() for root in self.roots]
        boards_states = torch.tensor(boards_states, device=self.model.device)
        policy, _ = self.model(boards_states)
        policy = policy.cpu().numpy()
        action_space_size = self.roots[0].board.size**4 if type(self.roots[0].board)==AttaxxBoard else self.roots[0].board.size**2+1
        dirichlet_arr = numpy.ones(shape=action_space_size)/action_space_size
        policy = (1-self.dirichlet_eps)*policy + self.dirichlet_eps*numpy.random.dirichlet(dirichlet_arr)
        # expand each root and associate the policy valyes with the ramifications
        for i in range(len(boards_obj_list)):
            self.roots[i].Expansion(policy[i])
            self.roots[i].n = 1

        # start MCTS iterations
        for _ in tqdm(range(self.n_iterations), desc="MCTS Iterations", leave=False, unit="iter", ncols=100, colour="#f7fc65"):             
            nodes = [self.roots[i].Select() for i in range(len(boards_obj_list))]
            # selection phase
            for i in range(len(boards_obj_list)):
                while len(nodes[i].children) > 0:
                    nodes[i] = nodes[i].Select()            
            # get the values from NN
            boards_states = [nodes[i].board.EncodedGameStateChanged() for i in range(len(boards_obj_list))]
            boards_states = torch.tensor(boards_states, device=self.model.device)
            policy, value = self.model(boards_states)
            policy = policy.cpu().numpy()
            value = value.cpu().numpy()
            for i in range(len(boards_obj_list)):
                # expansion phase
                nodes[i].Expansion(policy[i])
                # backprop phase
                nodes[i].BackPropagation(value[i][0])
        
        # return the actions' probabilities for each game
        action_space_size = self.roots[0].board.size**4 if type(self.roots[0].board) == AttaxxBoard else (self.roots[0].board.size**2)+1
        boards_actions_probs = [numpy.zeros(shape=action_space_size) for _ in range(len(boards_obj_list))]
        for i in range(len(boards_obj_list)):
            for action, child in self.roots[i].children.items():
                boards_actions_probs[i][action] = child.n
            boards_actions_probs[i] /= numpy.sum(boards_actions_probs[i])
        return boards_actions_probs

        # to obtain the resulting move
        # final_action = action_probs.argmax()
        # root.children[final_action].originMove
        