import numpy as np
from copy import deepcopy

class Group:
    def __init__(self, stone_coords, go_board):
        self.go_board = go_board
        self.stones = set([stone_coords])
        self.possible_prisioner = False

        neighbor_groups = set()
        i,j = stone_coords
        for (di, dj) in [(-1,0),(1,0),(0,-1),(0,1)]:
            if go_board.is_in(i+di, j+dj) and go_board.board[i+di, j+dj] == go_board.player:
                neighbor_groups.add(go_board.group_of_stone[(i+di, j+dj)])
        # updates the groups
        for neighbor_group in neighbor_groups:
            self.stones = self.stones.union(neighbor_group.stones)
            go_board.players_groups[go_board.player].remove(neighbor_group)
        for stone in self.stones:
            go_board.group_of_stone[stone] = self
        self.go_board.board[stone_coords] = self.go_board.player

        self.liberty_points = self.get_liberty_points()
        self.liberties = len(self.liberty_points)

    def get_liberty_points(self):
        liberty_pts = set()
        for (stone_i, stone_j) in self.stones:
            for (di, dj) in [(-1,0),(1,0),(0,-1),(0,1)]:
                if (self.go_board.is_in(stone_i+di, stone_j+dj)) and (self.go_board.board[stone_i+di, stone_j+dj] == 0):
                    liberty_pts.add((stone_i+di, stone_j+dj))
        return liberty_pts
    
    def count_liberties(self):
        return len(list(self.liberty_points))
    
    def __eq__(self, other):
        return self.stones.__eq__(other.stones)

    def __hash__(self):
        return object.__hash__(self)



class GoBoard:
    def __init__(self, dim):
        self.size = dim
        self.board = np.zeros(shape=(dim,dim), dtype=np.float32)
        self.player = 1
        self.players_groups = {1:[], 2:[]} # each dict holds key-value pairs of structure g:set( (i,j) ) | i,j integers, g Group
        self.group_of_stone = {(i,j): None for j in range(dim) for i in range(dim)}
        self.players_prev_boards = {1:None, 2:None}
        self.players_captured_opp_stones = {1:0, 2:0}
        self.players_last_move = {1:(-2,-2), 2:(-2,-2)}
        self.winner = 0

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
    
    def Start(self, render=True):
        if render:
            pygame.init()
            graphics.SET_GLOBALS("g",self.board)
            graphics.SET_SCREEN()
    
    def is_in(self, i, j):
        return 0 <= i < self.size and 0 <= j < self.size
    
    def NextPlayer(self):
        self.player = 3 - self.player

    def ValidMove(self, i, j):
        coords = (i,j)
        # pass play
        if coords == (-1,-1):
            return True
        if self.is_in(i,j):
            if self.board[coords] != 0:
                return False
            copy = deepcopy(self)
            Group(coords, copy)
            copy.players_groups[copy.player].append(copy.group_of_stone[coords])
            copy.update_player_groups()
            if copy.group_of_stone[coords].count_liberties() != 0:
                # repetition of board
                return (
                    (self.players_prev_boards[self.player] == None) or 
                    (self.players_prev_boards[self.player] != None and self.players_prev_boards[self.player] != str(copy.board))
                )
            # new group has zero liberties
            return False
        # out of bounds
        return False
                
    def update_player_groups(self):
        opponent_groups = self.players_groups[3-self.player]
        for group in opponent_groups:
            group.liberty_points = group.get_liberty_points()
            group.liberties = group.count_liberties()
            if group.liberties == 0:
                self.players_captured_opp_stones[self.player] += len(group.stones)
                for stone in group.stones:
                    self.board[stone] = 0
                    self.group_of_stone[stone] = None
                opponent_groups.remove(group)
        for group in self.players_groups[self.player]:
            group.liberty_points = group.get_liberty_points()
            group.liberties = group.count_liberties()

    def Move(self, coords):
        if self.ValidMove(*coords):
            self.players_last_move[self.player] = coords
            if coords != (-1,-1):
                Group(coords, self)
                self.players_groups[self.player].append(self.group_of_stone[coords])
                self.update_player_groups()
                self.players_prev_boards[self.player] = str(self.board)
            elif self.hasFinished():
                scores = self.calculate_scores()
                self.winner = 1 if scores[1] > scores[2] else 2
        else:
            print("NOT A VALID MOVE")

    def hasFinished(self):
        return (self.players_last_move[1] == (-1,-1) and self.players_last_move[2] == (-1,-1)) or self.winner != 0
    
    def PossibleMoves(self):
        moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i,j] == 0:
                    moves.append((i,j))
        moves.append((-1,-1))
        return moves
    
    def MoveToAction(self, move):
        if move == (-1,-1):
            return self.board.size**2
        i, j = move
        return j + i*self.board.size

    def calculate_scores(self):
        self_copy = deepcopy(self)
        board = deepcopy(self_copy.board) # copy the board

        def flood_fill(coords, player):
            if board[coords] == 0 or (board[coords] != 1 and board[coords] != 2): board[coords] += 10+player
            i,j = coords
            for (di, dj) in [(-1,0),(1,0),(0,-1),(0,1)]:
                if self_copy.is_in(i+di, j+dj) and board[i+di, j+dj] != 1 and board[i+di, j+dj] != 2 and board[i+di, j+dj] != 10+player and board[i+di, j+dj] < 15:
                    flood_fill((i+di, j+dj), player)
            return board
        
        # check for possible prisioners prisioners
        for player in [self_copy.player, 3-self_copy.player]:
            for opp_group in self_copy.players_groups[3-player]:
                board = deepcopy(self_copy.board) # copy the board
                for stone in opp_group.stones:
                    board = flood_fill(stone, 3-player)
                for group in self_copy.players_groups[player]:
                    # check if the player group liberties are inside the opponent territory
                    if all(map(lambda lib_pt: (board[lib_pt]==10+(3-player) or board[lib_pt]==(3-player)), group.liberty_points)):
                        group.possible_prisioner = True
        
        # remove prisioners
        board = deepcopy(self_copy.board) # copy the board
        for player in [self_copy.player, 3-self_copy.player]:
            for opp_group in self_copy.players_groups[3-player]:
                if opp_group.possible_prisioner:
                    continue
                for stone in opp_group.stones:
                    board = flood_fill(stone, 3-player)
            for group in deepcopy(self_copy.players_groups[player]):
                # check if the player group liberties are inside the opponent territory
                if group.possible_prisioner:
                    for stone in group.stones:
                        self_copy.board[stone] = 0
                        self_copy.group_of_stone[stone] = None
                    self_copy.players_groups[player].remove(group)

        # calculate territories
        board = deepcopy(self_copy.board) # copy the board
        for player in [3-self_copy.player, self_copy.player]:
            for group in self_copy.players_groups[player]:
                for stone in group.stones:
                    board = flood_fill(stone, player)

        # scores
        scores = {1:0, 2:0}
        # count territoy
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i,j] == 11: scores[1] += 1
                if board[i,j] == 12: scores[2] += 1
        # removing the amount of stones captured by the opponent
        scores[1] -= self_copy.players_captured_opp_stones[2]
        scores[2] -= self_copy.players_captured_opp_stones[1]
        # komi
        scores[2] += 5.5
        return scores
    
    def EncodedGameState(self):
        encoded_state = np.stack(
            (self.board == 1, self.board == 2, self.board == 0)
        ).astype(np.float32)
        return encoded_state


###############################################
####              TEST GAME                ####
###############################################

import pygame
import graphics

if __name__ == "__main__":
    b = GoBoard(9)
    pygame.init()
    graphics.SET_GLOBALS("g", b.board)
    graphics.SET_SCREEN()
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
        b.Move((i,j))
        b.NextPlayer()
        graphics.draw_board(b.board)
    print(b.calculate_scores())

