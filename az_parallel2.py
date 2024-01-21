import numpy as np
from aMCTS_parallel import MCTSParallel, MCTS_Node
from ataxx import AtaxxBoard
from go import GoBoard

import random
#from tqdm import tqdm
from tqdm.notebook import tqdm

import torch
import torch.nn.functional as F
torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")

def flip(array):
    temp = np.copy(array[:,:,:,6])
    array[:,:,:,6] = array[:,:,:,0]
    array[:,:,:,0] = temp
    temp = np.copy(array[:,:,:,5])
    array[:,:,:,5] = array[:,:,:,1]
    array[:,:,:,1] = temp
    temp = np.copy(array[:,:,:,4])
    array[:,:,:,4] = array[:,:,:,2]
    array[:,:,:,2] = temp
    return array

# for the data augmentation process
def transformations(board_state, action_probs, outcome, gameType, fill_size=0):
    if gameType == 'G':
        side = board_state.size
        transf = []
        # Flip vertically    
        transf.append((board_state.flip_vertical().EncodedGameStateChanged(), np.append(np.flip(np.copy(action_probs)[:-1].reshape(side,side),0).flatten(),action_probs[-1]), outcome))                         # flip vertically
        # Rotate 90 degrees
        transf.append((board_state.rotate90(1).EncodedGameStateChanged(), np.append(np.rot90(np.copy(action_probs)[:-1].reshape(side,side),1).flatten(),action_probs[-1]), outcome))                            # rotate 90
        # Rotate 90 degrees and flip vertically
        transf.append((board_state.rotate90(1).flip_vertical().EncodedGameStateChanged(), np.append(np.rot90(np.flip(np.copy(action_probs)[:-1].reshape(side,side),1),0).flatten(),action_probs[-1]), outcome)) # rotate 90 and flip vertically
        # Rotate 180 degrees
        transf.append((board_state.rotate90(2).EncodedGameStateChanged(), np.append(np.rot90(np.copy(action_probs)[:-1].reshape(side,side),2).flatten(),action_probs[-1]), outcome))                            # rotate 180
        # Rotate 180 degrees and flip vertically
        transf.append((board_state.rotate90(2).flip_vertical().EncodedGameStateChanged(), np.append(np.rot90(np.flip(np.copy(action_probs)[:-1].reshape(side,side),1),0).flatten(),action_probs[-1]), outcome)) # rotate 180 and flip vertically
        # Rotate 270 degrees
        transf.append((board_state.rotate90(3).EncodedGameStateChanged(), np.append(np.rot90(np.copy(action_probs)[:-1].reshape(side,side),3).flatten(),action_probs[-1]), outcome))                            # rotate 270
        # Rotate 270 degrees and flip vertically
        transf.append((board_state.rotate90(3).flip_vertical().EncodedGameStateChanged(), np.append(np.rot90(np.flip(np.copy(action_probs)[:-1].reshape(side,side),1),0).flatten(),action_probs[-1]), outcome)) # rotate 270 and flip vertically
        return transf
    elif gameType == 'A':
        if fill_size==0:
            side = board_state.size
            transf = []
            # Flip vertically 
            transf.append((board_state.flip_vertical().EncodedGameStateChanged(), np.flip(flip(np.copy(action_probs).reshape(side,side,2,8)),0).flatten(), outcome))                                                 # flip vertically
            # Rotate 90 degrees
            transf.append((board_state.rotate90(1).EncodedGameStateChanged(), np.rot90(np.roll(np.copy(action_probs).reshape(side,side,2,8),-2,axis=3),1,(0,1)).flatten(), outcome))                                       # rotate 90
            # Rotate 90 degrees and flip vertically
            transf.append((board_state.rotate90(1).flip_vertical().EncodedGameStateChanged(), np.flip(flip(np.rot90(np.roll(np.copy(action_probs).reshape(side,side,2,8),-2,axis=3),1,(0,1))),0).flatten(), outcome)) # rotate 90 and flip vertically
            # Rotate 180 degrees
            transf.append((board_state.rotate90(2).EncodedGameStateChanged(), np.rot90(np.roll(np.copy(action_probs).reshape(side,side,2,8),-4,axis=3),2,(0,1)).flatten(), outcome))                                       # rotate 180
            # Rotate 180 degrees and flip vertically
            transf.append((board_state.rotate90(2).flip_vertical().EncodedGameStateChanged(), np.flip(flip(np.rot90(np.roll(np.copy(action_probs).reshape(side,side,2,8),-4,axis=3),2,(0,1))),0).flatten(), outcome)) # rotate 180 and flip vertically
            # Rotate 270 degrees
            transf.append((board_state.rotate90(3).EncodedGameStateChanged(), np.rot90(np.roll(np.copy(action_probs).reshape(side,side,2,8),2,axis=3),3,(0,1)).flatten(), outcome))                                       # rotate 270
            # Rotate 270 degrees and flip vertically
            transf.append((board_state.rotate90(3).flip_vertical().EncodedGameStateChanged(), np.flip(flip(np.rot90(np.roll(np.copy(action_probs).reshape(side,side,2,8),2,axis=3),3,(0,1))),0).flatten(), outcome)) # rotate 270 and flip vertically
            return transf
        else:
            side = board_state.size
            transf = []
            # Flip vertically 
            transf.append((board_state.flip_vertical().EncodedGameStateChanged(fill_size), np.pad(np.flip(np.flip(np.copy(action_probs).reshape(fill_size,fill_size,fill_size,fill_size)[:side,:side,:side,:side],2),0),(0,fill_size-side),'constant',constant_values=(0)).flatten(), outcome))                                                 # flip vertically
            # Rotate 90 degrees
            transf.append((board_state.rotate90(1).EncodedGameStateChanged(fill_size),np.pad(np.rot90(np.rot90(np.copy(action_probs).reshape(fill_size,fill_size,fill_size,fill_size)[:side,:side,:side,:side],1,(2,3)),1,(0,1)),(0,fill_size-side),'constant',constant_values=(0)).flatten(), outcome))                                       # rotate 90
            # Rotate 90 degrees and flip vertically
            transf.append((board_state.rotate90(1).flip_vertical().EncodedGameStateChanged(fill_size), np.pad(np.flip(np.flip(np.rot90(np.rot90(np.copy(action_probs).reshape(fill_size,fill_size,fill_size,fill_size)[:side,:side,:side,:side],1,(2,3)),1,(0,1)),2),0),(0,fill_size-side),'constant',constant_values=(0)).flatten(), outcome)) # rotate 90 and flip vertically
            # Rotate 180 degrees
            transf.append((board_state.rotate90(2).EncodedGameStateChanged(fill_size), np.pad(np.rot90(np.rot90(np.copy(action_probs).reshape(fill_size,fill_size,fill_size,fill_size)[:side,:side,:side,:side],2,(2,3)),2,(0,1)),(0,fill_size-side),'constant',constant_values=(0)).flatten(), outcome))                                       # rotate 180
            # Rotate 180 degrees and flip vertically
            transf.append((board_state.rotate90(2).flip_vertical().EncodedGameStateChanged(fill_size), np.pad(np.flip(np.flip(np.rot90(np.rot90(np.copy(action_probs).reshape(fill_size,fill_size,fill_size,fill_size)[:side,:side,:side,:side],2,(2,3)),2,(0,1)),2),0),(0,fill_size-side),'constant',constant_values=(0)).flatten(), outcome)) # rotate 180 and flip vertically
            # Rotate 270 degrees
            transf.append((board_state.rotate90(3).EncodedGameStateChanged(fill_size), np.pad(np.rot90(np.rot90(np.copy(action_probs).reshape(fill_size,fill_size,fill_size,fill_size)[:side,:side,:side,:side],3,(2,3)),3,(0,1)),(0,fill_size-side),'constant',constant_values=(0)).flatten(), outcome))                                       # rotate 270
            # Rotate 270 degrees and flip vertically
            transf.append((board_state.rotate90(3).flip_vertical().EncodedGameStateChanged(fill_size), np.pad(np.flip(np.flip(np.rot90(np.rot90(np.copy(action_probs).reshape(fill_size,fill_size,fill_size,fill_size)[:side,:side,:side,:side],3,(2,3)),3,(0,1)),2),0),(0,fill_size-side),'constant',constant_values=(0)).flatten(), outcome)) # rotate 270 and flip vertically
            return transf
    return []


# function that applies temperature to the given probabilities distribution and normalizes the result, for the current AlphaZero iteration
def probs_with_temperature(probabilities, az_iteration):
    # returns a vale between 1.25 and 0.75
    def temperature_function(az_iter):
        return 1 / (1 + np.e**(az_iter-5)) + 0.5
    prob_temp =  probabilities**(1/temperature_function(az_iteration))
    prob_temp /= np.sum(prob_temp)
    return prob_temp


class AlphaZeroParallel2:
    """
    Class implementing the AlphaZero algorithm with parallelized self-play, MCTS, and training.

    Parameters:
        - model: The neural network model.
        - optimizer: The optimizer used for training the neural network.
        - board: The game board.
        - gameType: Type of the game ('G' for Go, 'A' for Ataxx).
        - data_augmentation: Flag for enabling data augmentation during self-play.
        - verbose: Flag for printing progress information.
        - fill_size: The fill size (used for Ataxx game with a fill).
        - **params: Additional parameters for configuration.

    Methods:
        - SelfPlay: Perform self-play for a specified number of iterations.
        - Train: Train the neural network on a given dataset.
        - Learn: Execute the AlphaZero algorithm for a specified number of iterations.

    """
     
    def __init__(self, model, optimizer, board, gameType, data_augmentation=False, verbose=False, fill_size=0, **params):
        """
        Initialize the AlphaZeroParallel2 object.

        Parameters:
            - model: The neural network model.
            - optimizer: The optimizer used for training the neural network.
            - board: The game board.
            - gameType: Type of the game ('G' for Go, 'A' for Ataxx).
            - data_augmentation: Flag for enabling data augmentation during self-play.
            - verbose: Flag for printing progress information.
            - fill_size: The fill size (used for Ataxx game with a fill).
            - **params: Additional parameters for configuration.

        """
        self.model = model
        self.optimizer = optimizer
        self.board = board
        self.gameType = gameType
        self.params = params
        self.data_augmentation = data_augmentation
        self.verbose = verbose
        self.fill_size = fill_size

    def SelfPlay(self, az_iteration):
        """
        Perform self-play for a specified number of iterations.

        Parameters:
            - az_iteration: The current iteration of the AlphaZero algorithm.

        Returns:
            - return_dataset: A list of training samples (state, action probabilities, outcome).

        """
        # Set the size of the game board
        if self.fill_size != 0:
            size = self.fill_size
        else:
            size = self.board.size
        
        # Initialize the return dataset to store training samples
        return_dataset = []

        # Track the number of self-plays performed
        selfplays_done = 0

        # Create a list of game boards, each associated with a separate thread for parallel self-play
        boards = [None for _ in range(self.params["n_self_play_parallel"])]
        boards_dataset = [[] for _ in range(self.params["n_self_play_parallel"])]
        boards_play_count = [0 for _ in range(self.params["n_self_play_parallel"])]

        # Initialize game boards based on the specified game type (Ataxx or Go)
        for i in range(self.params["n_self_play_parallel"]):
            boards[i] = AtaxxBoard(size) if self.gameType == "A" else GoBoard(size)
            boards[i].Start(render=False)

            # Adjust the size if the fill_size parameter is set
            if self.fill_size != 0:
                size -= 1

        # Reset size for self-play iterations
        if self.fill_size != 0:
            size = self.fill_size

        # Initialize the MCTS object for parallel search
        self.mcts = MCTSParallel(self.model, fill_size=self.fill_size)
        root_boards = [MCTS_Node(board, fill_size=self.fill_size) for board in boards]

        # Main loop for self-play
        while len(boards) > 0:
            # Use MCTS to get action probabilities for each board
            boards_actions_probs = self.mcts.Search(root_boards, self.params["mcts_iterations"])

            # Iterate over boards in reverse order to safely remove boards
            for i in range(len(boards))[::-1]:
                action_probs = boards_actions_probs[i]

                # Append the current state, action probabilities, and player to the dataset
                boards_dataset[i].append((boards[i].copy(), action_probs, boards[i].player))

                # Choose an action based on the probabilities
                moves = list(range(len(action_probs)))
                action = np.random.choice(moves, p=action_probs)
                move = self.mcts.roots[i].children[action].originMove

                # Apply the selected move to the board
                boards[i].Move(move)
                boards[i].NextPlayer()
                boards[i].CheckFinish()
                boards_play_count[i] += 1

                # Update the new root (root is now the played child state)
                root_boards[i] = self.mcts.roots[i].children[action]
                root_boards[i].parent = None  # It is needed to "remove" / "delete" the parent state

                # Check if the move cap is reached or the game is finished
                if boards_play_count[i] >= self.params["move_cap"] and boards[i].winner == 0:
                    boards[i].winner = 3

                if boards[i].hasFinished():
                    # Append the final configuration to the dataset
                    boards_dataset[i].append((boards[i].copy(), action_probs, boards[i].player))

                    # Switch to the next player and append the state again
                    boards[i].NextPlayer()
                    boards_dataset[i].append((boards[i].copy(), action_probs, boards[i].player))

                    # Process the dataset and add training samples with outcomes
                    for board, action_probs, player in boards_dataset[i]:
                        if player == boards[i].winner:
                            outcome = 1
                        elif 3 - player == boards[i].winner:
                            outcome = -1
                        else:
                            outcome = 0

                        # Add the training sample to the return dataset
                        return_dataset.append((board.EncodedGameStateChanged(self.fill_size), action_probs, outcome))

                        # Data augmentation process (rotating and flipping the board)
                        if self.data_augmentation:
                            for transformed_data in transformations(board, action_probs, outcome, self.gameType, fill_size=self.fill_size):
                                return_dataset.append(transformed_data)

                    # Dynamic parallel self-play allocation
                    if selfplays_done >= self.params["self_play_iterations"] - self.params["n_self_play_parallel"]:
                        del boards[i]
                        del root_boards[i]
                        del boards_play_count[i]
                    else:
                        # Initialize a new game board for self-play
                        boards[i] = AtaxxBoard(size) if self.gameType == "A" else GoBoard(size)
                        boards[i].Start(render=False)
                        root_boards[i] = MCTS_Node(boards[i], fill_size=self.fill_size)
                        boards_dataset[i] = []
                        boards_play_count[i] = 0

                        # Adjust the size if the fill_size parameter is set
                        if self.fill_size != 0:
                            if (selfplays_done + 1) % (self.fill_size - 3) == 0:
                                size = self.fill_size
                            else:
                                size -= 1

                    selfplays_done += 1

                    # Print progress message
                    if selfplays_done % self.params["n_self_play_parallel"] == 0:
                        print("\nSELFPLAY:", selfplays_done * 100 // self.params["self_play_iterations"], "%")

        print("\nSELFPLAY: 100 %")
        return return_dataset

    
    def Train(self, dataset):
        """
        Train the neural network on a given dataset.

        Parameters:
            - dataset: The dataset for training.

        """
        random.shuffle(dataset)

    # Iterate over the dataset in batches
        for batch_index in range(0, len(dataset), self.params['batch_size']):
            # Extract a batch of samples from the dataset
            sample = dataset[batch_index: batch_index + self.params["batch_size"]]
            
            # Unzip the samples into separate lists for board_encoded, policy_targets, and value_targets
            board_encoded, policy_targets, value_targets = zip(*sample)
            
            # Convert the lists to NumPy arrays
            board_encoded, policy_targets, value_targets = np.array(board_encoded), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            # Convert NumPy arrays to PyTorch tensors and move them to the device (GPU, if available)
            board_encoded = torch.tensor(board_encoded, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            # Forward pass: Get the model predictions for policy and value
            out_policy, out_value = self.model(board_encoded)

            # Calculate policy loss using cross-entropy loss
            policy_loss = F.cross_entropy(out_policy, policy_targets)

            # Calculate value loss using mean squared error loss
            value_loss = F.mse_loss(out_value, value_targets)

            # Combine policy and value losses with a weight factor for policy loss
            loss = policy_loss * 0.1 + value_loss

            # Zero the gradients, perform backward pass, and update model parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def Learn(self):
        """
        Execute the AlphaZero algorithm for a specified number of iterations.

        """
        for az_iteration in tqdm(range(self.params["n_iterations"]), desc="AlphaZero Algorithm Iterations", leave=False, unit="iter", ncols=100, colour="#fc6a65"):
            # Set the model in evaluation mode during self-play
            self.model.eval()
            
            # Perform self-play to generate a dataset
            dataset = self.SelfPlay(az_iteration)
            
            # Set the model back in training mode for updating parameters
            self.model.train()
            
            # Iterate over the specified number of training epochs
            for epoch in tqdm(range(self.params["n_epochs"]), desc="Training Model", leave=False, unit="epoch", ncols=100, colour="#9ffc65"):
                # Train the model using the generated dataset
                self.Train(dataset)
            
            # Save the model and optimizer states after each iteration
            if self.fill_size == 0:
                torch.save(self.model.state_dict(), f"./Models/{str.upper(self.gameType)}{self.board.size}/{str.upper(self.gameType)}{self.board.size}_{az_iteration}.pt")
                torch.save(self.optimizer.state_dict(), f"./Optimizers/{str.upper(self.gameType)}{self.board.size}/{str.upper(self.gameType)}{self.board.size}_{az_iteration}_opt.pt")
            else:
                torch.save(self.model.state_dict(), f"./Models/{str.upper(self.gameType)}Flex/{str.upper(self.gameType)}Flex_{az_iteration}.pt")
                torch.save(self.optimizer.state_dict(), f"./Optimizers/{str.upper(self.gameType)}Flex/{str.upper(self.gameType)}Flex_{az_iteration}_opt.pt")