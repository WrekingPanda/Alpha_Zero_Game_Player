import numpy 
from ataxx import AttaxxBoard
from go import GoBoard
import torch
torch.manual_seed(0)

#from tqdm import tqdm
from tqdm.notebook import tqdm


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
        
        #print("\nNEW NODE")
        for action, child in self.children.items():
            # Calculate UCB for each children 
            if child.n == 0:
                ucb = child.p*c*(self.n**(1/2))/(1+child.n)
            else:
                ucb = 1-((child.w/child.n)+1)/2 + child.p*c*(self.n**(1/2))/(1+child.n)
            # if child.n != 0:
            #     print("\n", 1-((child.w/child.n)+1)/2)
            # print(child.p*c*(self.n**(1/2))/(1+child.n))
            # print(child.p)
            # print()

            # Update max UCB value, as well as best Node
            if ucb > max_ucb: 
                max_ucb = ucb
                best_node = [child]
            elif ucb == max_ucb:
                best_node.append(child)
        return numpy.random.choice(best_node)

    def Expansion(self, policy:numpy.ndarray):
        if len(self.children) != 0 or self.board.winner != 0:
            return
        possibleMoves = self.board.PossibleMoves()
        masked_normalized_policy = numpy.zeros(shape=policy.shape)
        for move in possibleMoves:
            action = self.board.MoveToAction(move)
            masked_normalized_policy[action] = policy[action]
        masked_normalized_policy /= numpy.sum(masked_normalized_policy)

        for move in possibleMoves:
            cur_board = self.board.copy()
            cur_board.Move(move)
            action = cur_board.MoveToAction(move)
            cur_board.NextPlayer()
            cur_board.CheckFinish()
            self.children[action] = MCTS_Node(cur_board, self, move, policy_value=masked_normalized_policy[action])

    def BackPropagation(self, value):
        self.w += value
        self.n += 1
        if self.parent is not None:
            self.parent.BackPropagation(-1*value)


class MCTSParallel:
    def __init__(self, model) -> None:
        self.model = model
        self.roots = []

    @torch.no_grad()
    def Search(self, root_boards:list[MCTS_Node], n_iterations, dirichlet_eps=0.25, init_temp=None):
        self.roots = root_boards

        # add noise to the roots' policy array
        boards_states = [root.board.EncodedGameStateChanged() for root in self.roots]
        boards_states = torch.tensor(boards_states, device=self.model.device)
        policy, _ = self.model(boards_states)
        policy = policy.cpu().numpy()
        action_space_size = self.roots[0].board.size**4 if type(self.roots[0].board)==AttaxxBoard else self.roots[0].board.size**2+1
        dirichlet_arr = numpy.ones(shape=action_space_size)/action_space_size
        policy = (1-dirichlet_eps)*policy + dirichlet_eps*numpy.random.dirichlet(dirichlet_arr)
        # expand each root and associate the policy valyes with the ramifications
        for i in range(len(root_boards)):
            self.roots[i].Expansion(policy[i])
            self.roots[i].n = 1

        # start MCTS iterations
        for _ in tqdm(range(n_iterations), desc="MCTS Iterations", leave=False, unit="iter", ncols=100, colour="#f7fc65"):          
            nodes = [self.roots[i].Select() for i in range(len(root_boards))]
            # selection phase
            for i in range(len(root_boards)):
                while len(nodes[i].children) > 0:
                    nodes[i] = nodes[i].Select()
            # get the values from NN
            boards_states = [nodes[i].board.EncodedGameStateChanged() for i in range(len(root_boards))]
            boards_states = torch.tensor(boards_states, device=self.model.device)
            policy, value = self.model(boards_states)
            policy = policy.cpu().numpy()
            value = value.cpu().numpy()
            for i in range(len(root_boards)):
                if root_boards[i].board.winner!=0:
                    if root_boards[i].board.winner==root_boards[i].board.player:
                        nodes[i].BackPropagation(1)
                    elif root_boards[i].board.winner==3-root_boards[i].board.player:
                        nodes[i].BackPropagation(-1)
                else:
                    # expansion phase
                    nodes[i].Expansion(policy[i])
                    # backprop phase
                    nodes[i].BackPropagation(value[i][0].item())


        # print("\nROOT CHILDREN VISITS")
        # for root_board in root_boards:
        #     for child in root_board.children.values():
        #         print(child.originMove, child.n)
        #     print()
        
        # return the actions' probabilities for each game
        action_space_size = self.roots[0].board.size**4 if type(self.roots[0].board) == AttaxxBoard else (self.roots[0].board.size**2)+1
        boards_actions_probs = [numpy.zeros(shape=action_space_size) for _ in range(len(root_boards))]
        for i in range(len(root_boards)):
            for action, child in self.roots[i].children.items():
                boards_actions_probs[i][action] = child.n
            boards_actions_probs[i] /= numpy.sum(boards_actions_probs[i])
        return boards_actions_probs

        # to obtain the resulting move
        # final_action = action_probs.argmax()
        # root.children[final_action].originMove
        