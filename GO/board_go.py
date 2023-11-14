import numpy as np

EMPTY = 0
BLACK = 1
WHITE = 2
LIBERTY = 8

class board_go:
    def __init__(self, size) -> None:
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.player = BLACK # black is first to play
        self.black_pass_count = 0
        self.white_pass_count = 0

    
    #########################################################################################
    # PRINT #
    #########################################################################################

    def Print(self):
        s = "  "
        for i in range(self.size):
            s += "  " + str(i)
        print(s)
        
        line = "  +"
        for i in range(self.size * 3):
            line += "-"
        line += "+" 
        print(line)

        for i in range(self.size):
            row = str(i) + " |"
            for j in range(self.size):
                if(self.board[i][j] == EMPTY): 
                    row += " . "
                elif(self.board[i][j] == BLACK): 
                    row += " X "
                elif(self.board[i][j] == WHITE): 
                    row += " O "
            row += "|"    
            print(row)

        print(line)


    #########################################################################################
    # PLAYER TURN #
    #########################################################################################

    # changes curent player of the board 
    def ChangePLayer(self):
        if self.player == BLACK:
            self.player = WHITE
        else:
            self.player = BLACK

    # returns true if (row, col) is inside the board 
    def is_inside(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size

    # returns true when (row, col) is surrounded by oponents pieces
    def is_surrounded(self, row, col):
        count = 0
        pos = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for i in range(len(pos)):
            new_row = row + pos[i][0]
            new_col = col + pos[i][1]
            if self.is_inside(new_row, new_col) and self.board[new_row][new_col] != abs(self.player - 3):
                count += 1

        if count > 0:
            return False 
        else:
            return True

    # returns true if (row, col) is inside the board, is not surrounded and is empty
    def is_valid_move(self, row, col):
        return self.is_inside(row, col) and not(self.is_surrounded(row, col)) and self.board[row][col] == 0


    # places a piece if the move (row, col) is valid
    def PutPiece(self, row, col):
        # if move is valid, make the move
        if self.is_valid_move(row, col):
            # set piece in the board
            self.board[row][col] = self.player

            # restore pass count to 0, if a player makes a move
            if self.player == 1:
                self.black_pass_count = 0 
            else:
                self.white_pass_count = 0

            # changing player after piece placement 
            self.ChangePLayer()
            return True
        else:
            return False

    # a player may choose to pass their turn
    def PassTurn(self):
        if self.player == BLACK:
            self.black_pass_count += 1
            print("Black passed his turn!")
        else:
            self.white_pass_count += 1
            print("White passed his turn!")

        self.ChangePLayer()


    #########################################################################################
    # LIBERTIES COUNT AND REMOVE PHASE #
    #########################################################################################

    # Counting Liberties
    def count_liberties(self, row, col, visited=None):
        if visited is None:
            visited = set()

        # Check if the current position is out of bounds or already visited
        if not (0 <= row < self.size and 0 <= col < self.size) or (row, col) in visited:
            return 0

        visited.add((row, col))

        # Check if the current position is empty, in which case it contributes a liberty
        if self.board[row][col] == 0:
            return 1

        # Check the neighboring positions recursively
        liberties = 0
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row = row + dr
            new_col = col + dc
            liberties += self.count_liberties(new_row, new_col, visited)

        return liberties

    def count_group_liberties(self, row, col):
        stone_color = self.board[row][col]
        if stone_color == EMPTY:
            return 0  # Empty intersection has no liberties

        visited = set()
        return self.count_liberties(row, col, visited)
    

    #########################################################################################
    # GAME OVER ? #
    #########################################################################################

    def IsGameOver(self):
        if self.black_pass_count == 2:
            # white wins
            pass
        elif self.white_pass_count == 2:
            # black wins 
            pass



board = board_go(7)


board.PutPiece(0,0)
board.PutPiece(0,1)

board.PutPiece(5,6)
board.PutPiece(6,6)

print(board.count_group_liberties(6,6))
board.Print()
