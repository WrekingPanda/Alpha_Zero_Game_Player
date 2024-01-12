import numpy 
from ataxx import AttaxxBoard
from fastgo import GoBoard
import torch
torch.manual_seed(0)

#from tqdm import tqdm
from tqdm.notebook import tqdm

class MCTS_Node:
    def __init__(self, board, parent=None, move=None, policy_value=0, fill_size=0) -> None:
        # Initialize MCTS_Node with the given parameters
        self.board = board
        self.w = 0  # Sum of backpropagation
        self.n = 0  # Num of visits
        self.p = policy_value  # Probability returned from NN

        self.originMove = move
        self.parent = parent
        self.children = {}  # Save all children
        self.fill_size = fill_size

    def Select(self):
        # UCB constant
        c = 2
        max_ucb = -numpy.inf
        best_node = []

        # If the node has no children, return itself
        if len(self.children) == 0:
            return self

        # Iterate through children and calculate UCB for each
        for action, child in self.children.items():
            if child.n == 0:
                ucb = child.p * c * (self.n ** (1 / 2)) / (1 + child.n)
            else:
                ucb = -(child.w / child.n) + child.p * c * (self.n ** (1 / 2)) / (1 + child.n)

            # Update max UCB value, as well as the best Node
            if ucb > max_ucb:
                max_ucb = ucb
                best_node = [child]
            elif ucb == max_ucb:
                best_node.append(child)

        # Randomly choose one of the best nodes
        return numpy.random.choice(best_node)

    def Expansion(self, policy: numpy.ndarray):
        # Expand the node by creating child nodes for possible moves
        if len(self.children) != 0 or self.board.winner != 0:
            return

        # Mask the policy with valid moves
        masked_normalized_policy = numpy.zeros(shape=policy.shape)
        possible_moves = list(self.board.PossibleMoves())

        for move in possible_moves:
            action = self.board.MoveToAction(move, self.fill_size)
            masked_normalized_policy[action] = policy[action]

        # Normalize the masked policy
        if numpy.sum(masked_normalized_policy) != 0 and not numpy.isnan(numpy.sum(masked_normalized_policy)):
            masked_normalized_policy /= numpy.sum(masked_normalized_policy)
        else:
            # If there is any issue with the probabilities, distribute uniform probabilities
            masked_normalized_policy = numpy.ones(len(masked_normalized_policy)) / len(possible_moves)

        # Create child nodes for each possible move
        for move in possible_moves:
            cur_board = self.board.copy()
            cur_board.Move(move)
            action = cur_board.MoveToAction(move)
            cur_board.NextPlayer()
            cur_board.CheckFinish()
            self.children[action] = MCTS_Node(cur_board, self, move, policy_value=masked_normalized_policy[action],
                                              fill_size=self.fill_size)

    def BackPropagation(self, value):
        # Backpropagate the value up the tree
        cur = self
        while cur is not None:
            cur.w += value
            cur.n += 1
            value *= -1
            cur = cur.parent


class MCTSParallel:
    def __init__(self, model, fill_size=0) -> None:
        # Initialize MCTSParallel with the given parameters
        self.model = model
        self.roots = []
        self.fill_size = fill_size

    @torch.no_grad()
    def Search(self, root_boards: list[MCTS_Node], n_iterations, test=False):
        # Perform MCTS search for a specified number of iterations
        self.roots = root_boards

        # Add noise to the roots' policy array
        boards_states = [root.board.EncodedGameStateChanged(self.fill_size) for root in self.roots]
        boards_states = torch.tensor(boards_states, device=self.model.device)
        policy, _ = self.model(boards_states)
        policy = policy.cpu().numpy()

        # Calculate action space size based on game type and fill size
        if self.fill_size == 0:
            action_space_size = self.roots[0].board.size ** 4 if type(self.roots[0].board) == AttaxxBoard else \
                (self.roots[0].board.size ** 2) + 1
        else:
            action_space_size = self.fill_size ** 4

        # Expand each root and associate the policy values with the ramifications
        for i in range(len(root_boards)):
            if not test:
                possible_moves = self.roots[i].board.PossibleMoves()
                masked_normalized_policy = numpy.zeros(shape=policy[i].shape)

                # Mask the policy with valid moves
                for move in possible_moves:
                    action = self.roots[i].board.MoveToAction(move)
                    masked_normalized_policy[action] = policy[i][action]

                # Normalize the masked policy
                masked_normalized_policy /= numpy.sum(masked_normalized_policy)
                noise = numpy.random.normal(0, 0.01, len(masked_normalized_policy))
                policy[i] = numpy.abs(masked_normalized_policy + noise)

            self.roots[i].Expansion(policy[i])
            self.roots[i].n += 1

        # Start MCTS iterations
        for _ in tqdm(range(n_iterations - min([root.n for root in self.roots]) + 1), desc="MCTS Iterations",
                      leave=False, unit="iter", ncols=100, colour="#f7fc65"):
            nodes = [self.roots[i].Select() for i in range(len(root_boards))]

            # Selection phase
            for i in range(len(root_boards)):
                while len(nodes[i].children) > 0:
                    nodes[i] = nodes[i].Select()

            # Get values from NN
            boards_states = [nodes[i].board.EncodedGameStateChanged(self.fill_size) for i in range(len(root_boards))]
            boards_states = torch.tensor(boards_states, device=self.model.device)
            policy, value = self.model(boards_states)
            policy = policy.cpu().numpy()
            value = value.cpu().numpy()

            for i in range(len(root_boards)):
                if nodes[i].board.winner != 0:
                    # If the game has already finished, update the backpropagation value
                    if nodes[i].board.winner == nodes[i].board.player:
                        nodes[i].BackPropagation(1)
                    elif nodes[i].board.winner == 3 - nodes[i].board.player:
                        nodes[i].BackPropagation(-1)
                    else:
                        nodes[i].BackPropagation(0)
                else:
                    # Expansion phase
                    nodes[i].Expansion(policy[i])
                    # Backpropagation phase
                    nodes[i].BackPropagation(value[i][0].item())

        # Return the actions' probabilities for each game
        if self.fill_size == 0:
            action_space_size = self.roots[0].board.size ** 4 if type(self.roots[0].board) == AttaxxBoard else \
                (self.roots[0].board.size ** 2) + 1
        else:
            action_space_size = self.fill_size ** 4

        boards_actions_probs = [numpy.zeros(shape=action_space_size) for _ in range(len(root_boards))]
        for i in range(len(root_boards)):
            for action, child in self.roots[i].children.items():
                boards_actions_probs[i][action] = child.n
            boards_actions_probs[i] /= numpy.sum(boards_actions_probs[i])

        return boards_actions_probs