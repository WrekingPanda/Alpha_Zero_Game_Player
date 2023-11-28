import numpy as np
import random
import graphics
import pygame
from time import sleep

class AttaxxBoard:
    def __init__(self, dim):
        self.size = dim
        self.board = np.zeros((self.size,self.size),dtype=int)
        self.player = 1
        self.winner = 0
        
    def Start(self):
        self.board[0][0] = 1
        self.board[-1][-1] = 1
        self.board[0][-1] = 2
        self.board[-1][0] = 2
        pygame.init()
        graphics.SET_GLOBALS("a",self.board)
        graphics.SET_SCREEN()

    
    def ShowBoard(self, filling = False):
        if not(filling):
            print(f"Player: {self.player}")
        for i in range(self.size):
            line = ""
            for j in range(self.size):
                line += str(self.board[i][j]) + " "
            print(line)
        print()

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
        if nextRow==row and nextCol==col:
            return False
        if self.board[nextRow,nextCol] != 0:    # check if the next place if free to move too
            return False
        if nextRow==row and nextCol==col:   # check if the play is staying on the same place
            return False
        if abs(nextRow-row) > 2 or abs(nextCol-col) > 2:
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

    def Fill(self):
        graphics.unselect_piece()
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    self.board[i][j] = 3- self.player
                    self.ShowBoard(True)
                    graphics.draw_board(self.board)
                    pygame.display.flip()
                    sleep(1/(2*self.size))
    
    def PieceCount(self):
        count1= 0
        count2= 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 1:
                    count1+=1
                elif self.board[i][j] == 2:
                    count2+=1
        return count1,count2
    
    def CheckFinish(self):
        if (len(self.PossibleMoves())) == 0:
            self.Fill()
            c1,c2 = self.PieceCount()
            if c1 > c2:
                self.winner = 1
            elif c1 < c2:
                self.winner = 2
            else:
                self.winner = 3
    

def GameLoop():
    size = int(input("Size: "))
    board = AttaxxBoard(size)
    board.Start()
    graphics.draw_board(board.board)
    pygame.display.flip()
    board.ShowBoard()
    while board.winner==0:
        #a,b,c,d = list(map(int, input("Move:").split(' ')))
        #move = [a,b,c,d]
        graphics.show_piece_place()
        selected=False
        while(not selected):
            y1,x1 = graphics.piece_index_click()
            while board.board[y1][x1] != board.player:
                print("Invalid Position")
                y1,x1 = graphics.piece_index_click()
            graphics.set_selected_piece(y1,x1)
            graphics.draw_board(board.board)
            pygame.display.flip()
            y2,x2 = graphics.piece_index_click()
            if not (board.ValidMove(y1,x1,y2,x2)):
                print("Invalid Move")
                graphics.unselect_piece()
                graphics.draw_board(board.board)
                pygame.display.flip()
            else:
                selected = True
        board.Move([y1,x1,y2,x2])
        board.NextPlayer()
        graphics.unselect_piece()
        board.ShowBoard()
        graphics.draw_board(board.board)
        pygame.display.flip()
        board.CheckFinish()
    if board.winner == 3:
        print("Empate")
    else:
        print(f"Player {board.winner} wins")
    
    #####################
    # Test
    graphics.game_over(board.winner,board.board)
    graphics.show_pieces_amount()
    pygame.display.flip()
    pygame.time.wait(5000)
    
GameLoop()