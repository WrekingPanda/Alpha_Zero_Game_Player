import numpy as np
import random

class AttaxxBoard:
    def __init__(self, dim):
        self.size = dim
        self.board = np.zeros((self.size,self.size),dtype=int)
        self.player = 1
        
    def Start(self):
        self.board[0][0] = 1
        self.board[-1][-1] = 1
        self.board[0][-1] = 2
        self.board[-1][0] = 2
    
    def testStart(self):
        self.board[int(self.size/2)][int(self.size/2)] = 1
        self.board[int(self.size/2)][int(self.size/2) + 2] = 2
    
    def ShowBoard(self):
        for i in range(self.size):
            line = ""
            for j in range(self.size):
                line += str(self.board[i][j]) + " "

            print(line)

    def PossibleMoves(self):
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == self.player:
                    for nextRow in range(row-2,row+3):
                        for nextCol in range(col-2,col+3):
                            if self.ValidMove(row,col,nextRow,nextCol):
                                moves.append([row,col,nextRow,nextCol])
        return moves
    
    def ValidMove(self, row, col, nextRow, nextCol):
        if nextRow<0 or nextRow>=self.size or nextCol<0 or nextCol>=self.size:    # check if the next position is within the board
            return False
        if self.board[nextRow,nextCol] != 0:    # check if the next place if free to move too
            return False
        if nextRow==row and nextCol==col:   # check if the play is staying on the same place
            return False
        if abs(nextRow-row) + abs(nextCol-col) == 3:    # check for the invalid moves on range 2
            return False
        return True

    def Move(self,moveList):
        x1,y1,x2,y2 = moveList
        if (abs(x2-x1)>1 or abs(y2-y1)>1):
            self.board[x1][y1] = 0
        self.board[x2][y2] = self.player
        self.CapturePieces(x2,y2)
    
    def CapturePieces(self,x,y):
        for x2 in range(x-1,x+2):
            for y2 in range(y-1,y+2):
                if not(x2<0 or x2>=self.size or y2<0 or y2>=self.size):
                    if self.board[x2,y2] == 3-self.player:
                        self.board[x2,y2] = self.player
    
    def NextPlayer(self):
        self.player = 3-self.player
    
    def CheckFinish(self):
        winner = -1
        if self.HasMoves() == 0:
            self.GameOver()

    def HasMoves(self):
        state = 0
        store_player = self.player
        self.player = 1
        if (len(self.PossibleMoves()) > 0):
            state += 1
        self.NextPlayer()
        if (len(self.PossibleMoves()) > 0):
            state += 2
        self.player = store_player
        return state

    

def GameLoop():
    size = int(input("Size: "))
    board = AttaxxBoard(size)
    #board.Start()
    board.testStart()
    board.ShowBoard()

    print()

    moves = board.PossibleMoves()
    move = [int(size/2),int(size/2), int(size/2), int(size/2) + 1]
    board.Move(move)
    board.ShowBoard()
    
GameLoop()