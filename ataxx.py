import numpy as np
import graphics
import pygame
from time import sleep

VECTOR_INDEX = {(2,-2):0,(2,0):1,(2,2):2,(0,2):3,(-2,2):4,(-2,0):5,(-2,-2):6,(0,-2):7,(1,-1):8,(1,0):9,(1,1):10,(0,1):11,(-1,1):12,(-1,0):13,(-1,-1):14,(0,-1):15}

class AtaxxBoard:
    def __init__(self, dim):
        # Initialize the game board
        self.size = dim
        self.board = np.zeros(shape=(self.size, self.size), dtype=int)
        self.player = 1
        self.winner = 0

    def copy(self):
        # Create a deep copy of the current board state
        copy = AtaxxBoard(self.size)
        copy.board = np.copy(self.board)
        copy.player = self.player
        copy.winner = self.winner
        return copy

    def Start(self, render=False):
        # Set up the initial pieces on the board
        self.board[0][0] = 1
        self.board[-1][-1] = 1
        self.board[0][-1] = 2
        self.board[-1][0] = 2
        if render:
            # Initialize pygame and set up graphics 
            pygame.init()
            graphics.SET_GLOBALS("a", self.board)
            graphics.SET_SCREEN()

    def ShowBoard(self, filling=False):
        # Display the current state of the game board
        if not filling:
            print(f"Player: {self.player}")
        for i in range(self.size):
            line = ""
            for j in range(self.size):
                line += str(self.board[i][j]) + " "
            print(line)
        print()

    def PossibleMoves(self):
        # Generate all possible valid moves for the current player
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == self.player:
                    for nextRow in range(row - 2, row + 3):
                        for nextCol in range(col - 2, col + 3):
                            if self.ValidMove(row, col, nextRow, nextCol):
                                yield (row, col, nextRow, nextCol)

    def MoveToAction(self, move, fill_size=0):
        # Convert a move to an action index
        i1, j1, i2, j2 = move
        vector = (i2-i1,j2-j1)
        if fill_size == 0 or fill_size == self.size:
            size = self.size
            #action = j1 + i1 * size + j2 * (size ** 2) + i2 * (size ** 3)
            action = 16*(j1+i1*size) + VECTOR_INDEX[vector]
            return action
        else:
            action = j1 + i1 * fill_size + j2 * (fill_size ** 2) + i2 * (fill_size ** 3)
            return action

    def ValidMove(self, row, col, nextRow, nextCol):
        # Check if a move is valid
        if self.board[row, col] != self.player:
            return False
        if nextRow < 0 or nextRow >= self.size or nextCol < 0 or nextCol >= self.size:
            return False  # Check if the next position is within the board
        if nextRow == row and nextCol == col:
            return False  # Check if the play is staying on the same place
        if self.board[nextRow, nextCol] != 0:
            return False  # Check if the next place is free to move to
        if abs(nextRow - row) > 2 or abs(nextCol - col) > 2:
            return False
        if abs(nextRow - row) + abs(nextCol - col) == 3:
            return False  # Check for invalid moves on range 2
        return True

    def Move(self, moveList):
        # Make a move on the board
        x1, y1, x2, y2 = moveList
        if abs(x2 - x1) > 1 or abs(y2 - y1) > 1:
            self.board[x1][y1] = 0
        self.board[x2][y2] = self.player
        self.CapturePieces(x2, y2)

    def CapturePieces(self, x, y):
        # Capture opponent pieces surrounding the moved piece
        for x2 in range(x - 1, x + 2):
            for y2 in range(y - 1, y + 2):
                if not (x2 < 0 or x2 >= self.size or y2 < 0 or y2 >= self.size):
                    if self.board[x2, y2] == 3 - self.player:
                        self.board[x2, y2] = self.player

    def NextPlayer(self):
        # Switch to the next player's turn
        self.player = 3 - self.player

    def Fill(self, render=False):
        # Fill the empty spaces on the board with the opposite player's pieces
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    self.board[i][j] = 3 - self.player
                    if render:
                        self.ShowBoard(True)
                        graphics.unselect_piece()
                        graphics.draw_board(self.board)
                        pygame.display.flip()
                        sleep(1 / (2 * self.size))

    def PieceCount(self):
        # Count the number of pieces for each player
        count1 = 0
        count2 = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 1:
                    count1 += 1
                elif self.board[i][j] == 2:
                    count2 += 1
        return count1, count2

    def score(self, player):
        # Calculate the score difference between players
        points = self.PieceCount()
        if player == 1:
            return points[0] - points[1]
        return points[1] - points[0]

    def minimax(self, depth, max_player, alpha, beta, player):
        # Minimax algorithm with alpha-beta pruning for finding the best move
        self.CheckFinish()
        if self.winner == player:
            return None, self.size ** 2 + 1
        if self.winner == 3 - player:
            return None, -(self.size ** 2 + 1)
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

    def CheckFinish(self, render=False):
        # Check if the game has finished and determine the winner
        if len(list(self.PossibleMoves())) == 0:
            self.Fill(render)
            c1, c2 = self.PieceCount()
            if c1 > c2:
                self.winner = 1
            elif c1 < c2:
                self.winner = 2
            else:
                self.winner = 3

    def hasFinished(self):
        # Check if the game has finished
        return self.winner != 0

    def rotate90(self, times):
        # Rotate the game board 90 degrees clockwise
        copy = self.copy()
        copy.board = np.rot90(copy.board, times)
        return copy

    def flip_vertical(self):
        # Flip the game board vertically
        copy = self.copy()
        copy.board = np.flip(copy.board, 0)
        return copy

    def EncodedGameState(self):
        # Encode the current game state for input to a neural network
        encoded_state = np.stack(
            (self.board == 1, self.board == 2, self.board == 0)
        ).astype(np.float32)
        return encoded_state

    def EncodedGameStateChanged(self, fill_size=0):
        # Encode the game state depending on the current player for input to a neural network
        encoded_state = np.stack(
            (self.board == self.player, self.board == 3 - self.player, self.board == 0)
        ).astype(np.float32)
        if fill_size == 0 or fill_size == self.size:
            return encoded_state
        else:
            encoded = []
            for state in encoded_state:
                copy = np.copy(state)
                filled = np.pad(copy, (0, fill_size - self.size), 'constant', constant_values=(-1))
                encoded.append(filled)
            return np.array(encoded)


def GameLoop():
    # Get the size of the board from user input
    size = int(input("Size: "))
    
    # Initialize the game board
    board = AtaxxBoard(size)
    
    # Set up the initial game state and display
    board.Start(render=True)
    graphics.draw_board(board.board)
    pygame.display.flip()
    board.ShowBoard()
    
    # Main game loop
    while board.winner == 0:
        
        # Player's move input loop
        selected = False
        while not selected:
            # Get the starting position of the piece to move
            y1, x1 = graphics.piece_index_click()
            
            # Check if the selected position contains the player's piece
            while board.board[y1][x1] != board.player:
                print("Invalid Position")
                y1, x1 = graphics.piece_index_click()
            
            # Highlight the selected piece
            graphics.set_selected_piece(y1, x1)
            graphics.draw_board(board.board)
            pygame.display.flip()
            
            # Get the destination position for the move
            y2, x2 = graphics.piece_index_click()
            
            # Check if the move is valid
            if not board.ValidMove(y1, x1, y2, x2):
                print("Invalid Move")
                # Unselect the piece and redraw the board
                graphics.unselect_piece()
                graphics.draw_board(board.board)
                pygame.display.flip()
            else:
                selected = True
        
        # Make the move and update the game state
        board.Move([y1, x1, y2, x2])
        board.NextPlayer()
        graphics.unselect_piece()
        board.ShowBoard()
        graphics.draw_board(board.board)
        pygame.display.flip()
        
        # Check if the game has finished
        board.CheckFinish()

    # Display the game result
    if board.winner == 3:
        print("Tie")
    else:
        print(f"Player {board.winner} wins")

if __name__ == "__main__":
    GameLoop()