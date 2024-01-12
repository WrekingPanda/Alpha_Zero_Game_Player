import numpy as np

EMPTY = 0
BLACK = 1
WHITE = 2
MARKER = 4
LIBERTY = 8

class GoBoard:
    def __init__(self, dim):
        self.size = dim
        self.board = np.zeros(shape=(dim, dim), dtype=np.float32)
        self.player = BLACK
        self.winner = 0
        self.liberties = []
        self.group = []
        self.players_last_move = {1:(-2,-2), 2:(-2,-2)}
        self.players_prev_boards = {1:"", 2:""}

    def copy(self):
        copy = GoBoard(self.size)
        copy.board = self.board.copy()
        copy.player = self.player
        copy.winner = self.winner
        copy.liberties = self.liberties
        copy.group = self.group
        copy.players_last_move = self.players_last_move.copy()
        return copy

    def __eq__(self, other):
        return self.board.tolist() == other.board.tolist()

    def __str__(self):
        out = ""
        for _ in range(2*self.size+1): out += "-"
        out += "\n"
        for i in range(self.size):
            for j in range(-1, self.size+1):
                if j == -1 or j == self.size: out += "|"
                elif j == 0: out += str(self.board[i,j])
                else: out += " " + str(self.board[i,j])
            out += "\n"
        for _ in range(2*self.size+1): out += "-"
        return out
    
    def Start(self, render=False):
        if render:
            pygame.init()
            graphics.SET_GLOBALS("g", self.board)
            graphics.SET_SCREEN()
    
    def is_in(self, i, j):
        return 0 <= i < self.size and 0 <= j < self.size
    
    def NextPlayer(self):
        self.player = 3 - self.player

    def ValidMove(self, i, j):
        coords = (i,j)
        if coords == (-1,-1): return True
        if self.is_in(i,j):
            if self.board[coords] != EMPTY: return False
            if self.players_last_move[self.player] == coords: return False
            copy = self.copy()
            copy.Move(coords)
            # check for suicide move
            if copy.board[coords] == EMPTY: return False
            # if board is repetition of the last time the same player played, return False, otherwise return True
            return(
                (self.players_prev_boards[self.player] == "") or 
                (self.players_prev_boards[self.player] != "" and self.players_prev_boards[self.player] != str(copy.board))    
            )
        # out of bounds
        return False

    def CountLiberties(self, coords, player):
        if self.board[coords] == player:
            self.group.append(coords)
            self.board[coords] += MARKER
            i,j = coords
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                if self.is_in(i+di, j+dj):
                    self.CountLiberties((i+di, j+dj), player)
        elif self.board[coords] == EMPTY:
            self.board[coords] = LIBERTY
            self.liberties.append(coords)

    def ClearGroup(self):
        for coord in self.group:
            self.board[coord] = EMPTY
        self.group = []

    def RestoreBoard(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i,j] == LIBERTY: self.board[i,j] = EMPTY
                elif self.board[i,j] > MARKER: self.board[i,j] -= MARKER

    def CapturePieces(self, coords, opp_player):
        i,j = coords
        neighbor_cells = [(i+di, j+dj) for di, dj in [(-1,0), (1,0), (0,-1), (0,1)] if self.is_in(i+di, j+dj)]
        for neighbor_coords in neighbor_cells:
            if self.board[neighbor_coords] == opp_player:
                self.liberties = []
                self.group = []
                self.CountLiberties(neighbor_coords, opp_player)
                if len(self.liberties) == 0:
                    self.ClearGroup()
                self.RestoreBoard()

    def Move(self, coords):
        self.players_last_move[self.player] = coords
        if coords == (-1,-1):
            return
        self.board[coords] = self.player
        self.CapturePieces(coords, 3-self.player)
        self.players_prev_boards[self.player] = str(self.board)

    def CheckFinish(self, render=False):
        if (self.players_last_move[1] == (-1,-1) and self.players_last_move[2] == (-1,-1)):
            scores = self.CalculateScores()
            if scores[1] > scores[2]: self.winner = 1
            else: self.winner = 2 
        
    def hasFinished(self):
        return self.winner != 0
    
    def PossibleMoves(self):
        #moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.ValidMove(i,j):
                    yield (i,j)
                    #moves.append((i,j))
        yield (-1,-1)
        #moves.append((-1,-1))
        #return moves
    
    def MoveToAction(self, move, fill_size=0):
        if move == (-1,-1):
            return self.size**2
        i, j = move
        return j + i*self.size
    
    def score(self, player):
        points = self.CalculateScores()
        if player==1:
            return points[1]-points[2]
        return points[2]-points[1]
    
    def minimax(self,depth, max_player, alpha, beta, player):
        self.CheckFinish()
        if self.winner==player:
            return None,10000
        if self.winner==3-player:
            return None,-10000
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

    def CalculateScores(self):
        scores = {BLACK:0, WHITE:5.5}
        visited = set()

        def influence_score(i, j):
            score = 0
            for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                if self.is_in(i+di, j+dj):
                    score += self.board[i+di, j+dj]
            return score
        
        def explore_territory(i, j):
            nonlocal scores
            if (i,j) in visited or not self.is_in(i, j):
                return
            visited.add((i,j))
            if self.board[i,j] == 0:
                score = influence_score(i, j)
                if score > 0: scores[BLACK] += 1
                elif score < 0: scores[WHITE] += 1

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0 and (i, j) not in visited:
                    explore_territory(i, j)
        return scores
    
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
        return encoded_state


###############################################
####              TEST GAME                ####
###############################################

import pygame
import graphics

if __name__ == "__main__":
    b = GoBoard(9)
    b.Start(render=True)
    graphics.draw_board(b.board)
    count_pass = 0
    while True:
        i,j = graphics.piece_index_click()
        if (i,j) == (-1,-1):
            count_pass += 1
            if count_pass == 2:
                break
        else:
            count_pass = 0
        if b.ValidMove(i,j):
            b.Move((i,j))
            b.NextPlayer()
            b.CheckFinish()
        # else:
        #     print("NOT A VALID MOVE")
        graphics.draw_board(b.board)
    print(b.CalculateScores())
