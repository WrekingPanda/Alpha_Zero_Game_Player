import numpy 
from ataxx import AttaxxBoard
from copy import deepcopy

class MCTS_Node:
    def __init__(self, board, parent=None, move=None) -> None:
        self.board : AttaxxBoard = board
        self.w = 0 # Sum of backpropagation 
        self.n = 0 # Num of visits
        self.p = 0 # Probability returned from NN
        
        self.originMove = move
        self.parent = parent
        self.children = {} # Save all children 


    def Select(self):
        c = 2**(1/2)
        max_ucb = 0
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


    def Expantion(self): # Policy??? maybe we should change smth xd
        if len(self.children) != 0:
            return

        possibleMoves = self.board.possibleMoves()
        for move in possibleMoves:
            cur_board = deepcopy(self.board)
            cur_board.Move(move)

            self.children[move] = MCTS_Node(cur_board, self, move)


    def BackPropagation(self, value):
        self.w += value
        self.n += 1

        if self.parent is not None:
            self.parent.BackPropagation(-value)
           


class MCTS:
    def __init__(self, board, n_iterations, model) -> None:
        self.board = board
        self.n_iterations = n_iterations
        self.model = model 


    def Search(self):
        root = MCTS_Node(self.board)

        for i in range(self.n_iterations):
            node = root

            node = node.Select()
            policy, value = self.model() # Completeeee
            node.Expantion() # should pass policy hereee 
            node.BackPropagation(value)

        


      
        





