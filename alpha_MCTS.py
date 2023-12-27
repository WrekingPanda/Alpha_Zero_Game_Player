import numpy 
from ataxx import AttaxxBoard
from go import GoBoard
from copy import deepcopy

class MCTS_Node:
    def __init__(self, board, parent=None, move=None, policy_value=0) -> None:
        self.board : AttaxxBoard | GoBoard = board
        self.w = 0 # Sum of backpropagation 
        self.n = 0 # Num of visits
        self.p = policy_value # Probability returned from NN
        
        self.originMove = move
        self.parent: MCTS_Node = parent
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
                ucb = 0 # numpy.inf or child.p*c*(self.n**(1/2))/(1+child.n)

            # Update max UCB value, as well as best Node
            if ucb > max_ucb: 
                max_ucb = ucb
                best_node = [child]
            elif ucb == max_ucb:
                best_node.append(child)

        return numpy.random.choice(best_node)


    def Expansion(self): # Policy??? maybe we should change smth xd
        if len(self.children) != 0 or self.board.winner != 0:
            return

        possibleMoves = self.board.PossibleMoves()
        for move in possibleMoves:
            cur_board = deepcopy(self.board)
            cur_board.Move(move)
            action = self.board.MoveToAction(move)
            self.children[action] = MCTS_Node(cur_board, self, move)


    def BackPropagation(self, value):
        self.w += value
        self.n += 1
        if self.parent is not None:
            self.parent.BackPropagation(-1*value)
           


class MCTS:
    def __init__(self, board, n_iterations, model) -> None:
        self.board: AttaxxBoard | GoBoard = board
        self.n_iterations = n_iterations
        self.model = model

    def Search(self):
        root = MCTS_Node(self.board)

        for _ in range(self.n_iterations):
            node = root
            while len(node.children) > 0:
                node = node.Select()

            policy, value = self.model() # Completeeee

            node.Expansion() # should pass policy hereee 
            node.BackPropagation(value)
        
        action_space_size = self.board.size**2
        if type(self.board) == GoBoard:
            action_space_size += 1
        action_probs = numpy.zeros(shape=action_space_size)
        for action, child in root.children.items():
            action_probs[action] = child.n
        action_probs /= numpy.sum(action_probs)
        """ 
        if to return probabilities:
            return action_probs
        """
        final_action = action_probs.argmax()
        return root.children[final_action].originMove
