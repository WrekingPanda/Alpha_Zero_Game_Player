import pygame
import graphics
import numpy as np
import gogame


# "our" Go Implementation that uses the GymGo low level implementation

class GoBoard:
    def __init__(self, dim):
        self.size = dim
        self.player = 1
        self.winner = 0
        self.board = np.zeros(shape=(dim,dim), dtype=np.float64)
        self.state = gogame.init_state(dim)

    # Create a deep copy of the current board
    def copy(self):
        copy = GoBoard(self.size)
        copy.player = self.player
        copy.winner = self.winner
        copy.board = self.board
        copy.state = self.state
        return copy

    # String representation of the board for display purposes
    def __str__(self):
        return gogame.str(self.state)

    # Initialize the game board for rendering (using Pygame)
    def Start(self, render=False):
        if render:
            pygame.init()
            graphics.SET_GLOBALS("g", self.board)
            graphics.SET_SCREEN()

    # Check if a given coordinate is within the board boundaries
    def is_in(self, i, j):
        return 0 <= i < self.size and 0 <= j < self.size

    # Switch to the next player
    def NextPlayer(self):
        self.player = 3 - self.player

    # Check if a move is valid based on the game rules
    def ValidMove(self, i, j):
        action = self.MoveToAction((i,j))
        return bool(gogame.valid_moves(self.state)[action])

    # Perform a move and update the board state
    def Move(self, coords):
        action = self.MoveToAction(coords)
        self.state = gogame.next_state(self.state, action)
        self.board = np.zeros(shape=(self.size, self.size)) + self.state[0]*1 + self.state[1]*2

    # Check if the game has finished
    def CheckFinish(self, render=False):
        if bool(gogame.game_ended(self.state)):
            score = self.CalculateScores()
            if score > 0: self.winner = 1
            elif score < 0: self.winner = 2

    # Check if the game has finished
    def hasFinished(self):
        return self.winner != 0

    # Generate possible moves for the current player
    def PossibleMoves(self):
        valid_moves = gogame.valid_moves(self.state)
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                if bool(valid_moves[self.size * i + j]):
                    count += 1
                    yield (i, j)
        if count == 0 or np.sum(self.board == 0) < 5:
            yield (-1, -1)

    # Convert a move to an action index
    def MoveToAction(self, move, fill_size=0):
        if move == (-1, -1):
            return self.size ** 2
        i, j = move
        return j + i * self.size

    # Calculate scores based on the current board state
    def CalculateScores(self):
        return gogame.winning(self.state, komi=5.5)

    # Rotate the board 90 degrees clockwise 'times' times
    def rotate90(self, times):
        copy = self.copy()
        copy.board = np.rot90(copy.board, times)
        return copy
    
    # Flip the board vertically
    def flip_vertical(self):
        copy = self.copy()
        copy.board = np.flip(copy.board, 0)
        return copy

    # Encode the current game state for input to a neural network
    def EncodedGameState(self):
        encoded_state = np.stack(
            (self.board == 1, self.board == 2, self.board == 0)
        ).astype(np.float32)
        return encoded_state
    
    # Encode the game state depending on the current player for input to a neural network
    def EncodedGameStateChanged(self, fill_size=0):
        encoded_state = np.stack(
            (self.board == self.player, self.board == 3-self.player, self.board == 0)
        ).astype(np.float32)
        return encoded_state
    
    def score(self, player):
        # Calculate the score difference between players
        return self.CalculateScores()

    def minimax(self, depth, max_player, alpha, beta, player):
        # Minimax algorithm with alpha-beta pruning for finding the best move
        self.CheckFinish()
        if self.winner == player:
            return None, 10000
        if self.winner == 3 - player:
            return None, -10000
        if depth == 0:
            return None, self.score(player)
        if max_player:
            value = -np.inf
            for move in self.PossibleMoves():
                copy = self.copy()
                copy.Move(move)
                copy.NextPlayer()
                new = copy.minimax(depth - 1, False, alpha, beta, player)[1]
                if new > value:
                    value = new
                    best = move
                if value >= beta:
                    break
                alpha = max(alpha, value)
            return best, value
        else:
            value = np.inf
            for move in self.PossibleMoves():
                copy = self.copy()
                copy.Move(move)
                copy.NextPlayer()
                new = copy.minimax(depth - 1, True, alpha, beta, player)[1]
                if new < value:
                    value = new
                    best = move
                if value <= alpha:
                    break
                beta = min(beta, value)
            return best, value
        
def GameLoop():
    # Initialize Go board with dimension 9
    b = GoBoard(9)
    
    # Start the game rendering (using Pygame)
    b.Start(render=True)
    
    # Draw the initial board state
    graphics.draw_board(b.board)
    
    # Counter for consecutive passes
    count_pass = 0
    
    # Game loop
    while True:
        # Get user click coordinates
        i, j = graphics.piece_index_click()
        
        # Check for a pass move
        if (i, j) == (-1, -1):
            count_pass += 1
            if count_pass == 2:
                print("GAME ENDED --- SCORE:", b.CalculateScores())
                break
        else:
            count_pass = 0

        # Check if the move is valid
        if b.ValidMove(i, j):
            # Make the move, switch to the next player, and check for game finish
            b.Move((i, j))
            b.NextPlayer()
            b.CheckFinish()
        # Uncomment the else block to print a message for invalid moves
        # else:
        #     print("NOT A VALID MOVE")

        # Draw the updated board state
        graphics.draw_board(b.board)

if __name__ == "__main__":
    GameLoop()