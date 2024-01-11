import numpy as np
import graphics
import pygame
from time import sleep

class AttaxxBoard:
    def __init__(self, dim):
        self.size = dim
        self.board = np.zeros(shape=(self.size,self.size), dtype=int)
        self.player = 1
        self.winner = 0

    def copy(self):
        copy = AttaxxBoard(self.size)
        copy.board = np.copy(self.board)
        copy.player = self.player
        copy.winner = self.winner
        return copy
        
    def Start(self, render=False):
        self.board[0][0] = 1
        self.board[-1][-1] = 1
        self.board[0][-1] = 2
        self.board[-1][0] = 2
        if render:
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
        #moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == self.player:
                    for nextRow in range(row-2,row+3):
                        for nextCol in range(col-2,col+3):
                            if self.ValidMove(row,col,nextRow,nextCol):
                                yield (row,col,nextRow,nextCol)
                                #moves.append((row,col,nextRow,nextCol))
        #return moves
    
    def MoveToAction(self, move, fill_size=0):
        if fill_size==0 or fill_size==self.size:
            i1, j1, i2, j2 = move
            size = self.size
            action = j1 + i1*size + j2*(size**2) + i2*(size**3)
            return action
        else:
            i1, j1, i2, j2 = move
            action = j1 + i1*fill_size + j2*(fill_size**2) + i2*(fill_size**3)
            return action
    
    def ValidMove(self, row, col, nextRow, nextCol):
        if (self.board[row, col] != self.player):
            return False
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

    def Fill(self, render=False):
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    self.board[i][j] = 3-self.player
                    if render:
                        self.ShowBoard(True)
                        graphics.unselect_piece()
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
    
    def score(self, player):
        points = self.PieceCount()
        if player==1:
            return points[0]-points[1]
        return points[1]-points[0]
    
    def minimax(self,depth, max_player, alpha, beta, player):
        self.CheckFinish()
        if self.winner==player:
            return None,self.size**2+1
        if self.winner==3-player:
            return None,-(self.size**2+1)
        if depth==0:
            return None,self.score(player)
        if max_player:
            value = -np.inf
            for move in self.PossibleMoves():
                copy = self.copy()
                copy.Move(move)
                copy.NextPlayer()
                new = copy.minimax(depth-1,False,alpha,beta, player)[1]
                if new > value:
                    value = new
                    best = move
                if value >= beta:
                    break
                alpha = max(alpha,value)
            return best,value
        else:
            value = np.inf
            for move in self.PossibleMoves():
                copy = self.copy()
                copy.Move(move)
                copy.NextPlayer()
                new = copy.minimax(depth-1,True,alpha,beta, player)[1]
                if new < value:
                    value = new
                    best = move
                if value <= alpha:
                    break
                beta = min(beta,value)
            return best,value
    
    def CheckFinish(self, render=False):
        if (len(self.PossibleMoves())) == 0:
            self.Fill(render)
            c1,c2 = self.PieceCount()
            if c1 > c2:
                self.winner = 1
            elif c1 < c2:
                self.winner = 2
            else:
                self.winner = 3
    
    def hasFinished(self):
        return self.winner != 0
    
    def rotate90(self, times):
        copy = self.copy()
        copy.board = np.rot90(copy.board, times)
        return copy

    def flip_vertical(self):
        copy = self.copy()
        copy.board = np.flip(copy.board, 0)
        return copy
    
    def EncodedGameState(self):
        encoded_state = np.stack(
            (self.board == 1, self.board == 2, self.board == 0)
        ).astype(np.float32)
        return encoded_state

    def EncodedGameStateChanged(self, fill_size=0):
        encoded_state = np.stack(
            (self.board == self.player, self.board == 3-self.player, self.board == 0)
            ).astype(np.float32)
        if fill_size == 0 or fill_size==self.size:
            return encoded_state
        else:
            encoded = []
            for state in encoded_state:
                copy = np.copy(state)
                filled = np.pad(copy,(0,fill_size-self.size),'constant',constant_values=(-1))
                encoded.append(filled)
            return np.array(encoded)


def GameLoop():
    size = int(input("Size: "))
    board = AttaxxBoard(size)
    board.Start(render=True)
    graphics.draw_board(board.board)
    pygame.display.flip()
    board.ShowBoard()
    while board.winner==0:
        #a,b,c,d = list(map(int, input("Move:").split(' ')))
        #move = [a,b,c,d]
        graphics.show_pieces_amount()
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

if __name__ == "__main__":
    GameLoop()