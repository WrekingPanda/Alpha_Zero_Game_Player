import numpy 
from ataxx import AttaxxBoard
from go import GoBoard
from copy import deepcopy
import torch
from tqdm import tqdm

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
            if child.n != 0:
                ucb = (child.w/child.n) + child.p*c*(self.n**(1/2))/(1+child.n)
            else: 
                ucb = child.p*c*(self.n**(1/2))/(1+child.n) # exploration weight of the UCB formula

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
        policy = policy.squeeze(0)
        for move in possibleMoves:
            cur_board = deepcopy(self.board)
            cur_board.Move(move)
            action = cur_board.MoveToAction(move)
            cur_board.NextPlayer()
            self.children[action] = MCTS_Node(cur_board, self, move, policy_value=policy[action])


    def BackPropagation(self, value):
        self.w += value
        self.n += 1
        if self.parent is not None:
            self.parent.BackPropagation(-1*value)
           


class MCTS:
    def __init__(self, board, n_iterations, model) -> None:
        self.board = board
        self.n_iterations = n_iterations
        self.model = model
        self.root = MCTS_Node(self.board)

    @torch.no_grad()
    def Search(self):
        self.root = MCTS_Node(self.board)

        for _ in tqdm(range(self.n_iterations), desc="MCTS Iterations", leave=False, unit="iter", ncols=100, colour="#f7fc65"):
            node = self.root
            while len(node.children) > 0:
                node = node.Select()

            game_state = node.board.EncodedGameState()
            game_state = torch.tensor(game_state).unsqueeze(0)
            policy, value = self.model(game_state)

            node.Expansion(policy) 
            node.BackPropagation(value)
        
        action_space_size = self.board.size**4 if type(self.board) == AttaxxBoard else (self.board.size**2)+1
        action_probs = numpy.zeros(shape=action_space_size)
        #print(len(self.root.children.items()))
        for action, child in self.root.children.items():
            action_probs[action] = child.n
        action_probs /= numpy.sum(action_probs)
        return action_probs

        # to obtain the resulting move
        # final_action = action_probs.argmax()
        # root.children[final_action].originMove
        