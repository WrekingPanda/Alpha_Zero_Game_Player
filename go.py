import numpy as np
from copy import deepcopy

class Group:
    def __init__(self, stone_coords=None, go_board=None):
        # for copy method
        if stone_coords is None and go_board is None:
            self.stones = set()
            self.possible_prisioner = False
            self.liberty_points = set()
            self.liberties = 0
        # usual init behaviour
        else:
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
            go_board.board[stone_coords] = go_board.player

            self.liberty_points = self.get_liberty_points(go_board)
            self.liberties = len(self.liberty_points)

    def get_liberty_points(self, go_board):
        liberty_pts = set()
        for (stone_i, stone_j) in self.stones:
            for (di, dj) in [(-1,0),(1,0),(0,-1),(0,1)]:
                if (go_board.is_in(stone_i+di, stone_j+dj)) and (go_board.board[stone_i+di, stone_j+dj] == 0):
                    liberty_pts.add((stone_i+di, stone_j+dj))
        return liberty_pts
    
    def count_liberties(self):
        return len(list(self.liberty_points))
    
    def copy(self):
        copy = Group()
        copy.stones = self.stones.copy()
        copy.possible_prisioner = self.possible_prisioner
        copy.liberty_points = self.liberty_points.copy()
        copy.liberties = self.liberties
        return copy
    
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

    def copy(self):
        copy = GoBoard(self.size)
        copy.board = self.board.copy()
        copy.player = self.player
        copy.players_groups = {1:[], 2:[]}
        for p in [1,2]:
            for group in self.players_groups[p]:
                copy.players_groups[p].append(group.copy())
        copy.group_of_stone = {(i,j): None for j in range(self.size) for i in range(self.size)}
        for i in range(self.size):
            for j in range(self.size):
                if self.group_of_stone[(i,j)] is not None:
                    copy.group_of_stone[(i,j)] = self.group_of_stone[(i,j)].copy()
        copy.players_prev_boards = self.players_prev_boards.copy()
        copy.players_captured_opp_stones = self.players_captured_opp_stones.copy()
        copy.players_last_move = self.players_last_move.copy()
        copy.winner = self.winner
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
            # non-empty cell
            if self.board[coords] != 0:
                return False
            # repetition of board
            if self.players_last_move[self.player] == coords:
                return False
            copy = deepcopy(self)
            Group(coords, copy)
            copy.players_groups[copy.player].append(copy.group_of_stone[coords])
            copy.update_player_groups(coords)
            if copy.group_of_stone[coords].count_liberties() != 0:
                return (
                    (self.players_prev_boards[self.player] == None) or 
                    (self.players_prev_boards[self.player] != None and self.players_prev_boards[self.player] != str(copy.board))
                )
            # new group has zero liberties
            return False
        # out of bounds
        return False
                
    def update_player_groups(self, move):
        if move == (-1,-1):
            return

        played_stone = move
        adj_stones = [(played_stone[0]+di, played_stone[1]+dj) for di,dj in [(-1,0),(1,0),(0,-1),(0,1)] if self.is_in(played_stone[0]+di, played_stone[1]+dj)]
        updated_groups = set()
        for adj_stone in adj_stones:
            # stone of the same group
            if self.board[adj_stone] == self.board[played_stone] and self.group_of_stone[played_stone] not in updated_groups:
                #player group
                p_group = self.group_of_stone[played_stone]
                p_group.liberty_points = p_group.get_liberty_points(self)
                p_group.liberties = p_group.count_liberties()
                updated_groups.add(p_group)
            # stone of the opponent
            elif self.board[adj_stone] == 3-self.player:
                opp_group = self.group_of_stone[adj_stone]
                if opp_group not in updated_groups:
                    # opp group lost all liberties -> opp group to be deleted & player groups' liberties need to be updated
                    if len(opp_group.liberty_points) == 1 and played_stone in opp_group.liberty_points:
                        p_groups_to_be_updated = []
                        self.players_captured_opp_stones[self.player] += len(opp_group.stones)
                        for opp_stone in opp_group.stones:
                            # look for player's groups adjacent to the opp group to be deleted
                            opp_adj_stones = [(opp_stone[0]+di, opp_stone[1]+dj) for di,dj in [(-1,0),(1,0),(0,-1),(0,1)] if self.is_in(opp_stone[0]+di, opp_stone[1]+dj)]
                            for opp_adj_stone in opp_adj_stones:
                                if self.board[opp_adj_stone] == self.player:
                                    p_groups_to_be_updated.append(self.group_of_stone[opp_adj_stone])
                            # delete opp group
                            self.board[opp_stone] = 0
                            self.group_of_stone[opp_stone] = None
                        # update registed player's groups
                        for p_group in p_groups_to_be_updated:
                            p_group.liberty_points = p_group.get_liberty_points(self)
                            p_group.liberties = p_group.count_liberties()
                    # remove the liberty point
                    elif len(opp_group.liberty_points) > 1:
                        opp_group.liberty_points.remove(played_stone)
                        opp_group.liberties -= 1
                        updated_groups.add(opp_group)

    def Move(self, coords):
        self.players_last_move[self.player] = coords
        if coords != (-1,-1):
            Group(coords, self)
            self.players_groups[self.player].append(self.group_of_stone[coords])
            self.update_player_groups(coords)
            self.players_prev_boards[self.player] = str(self.board)
        elif self.hasFinished():
            scores = self.calculate_scores()
            self.winner = 1 if scores[1] > scores[2] else 2

    def CheckFinish(self):
        if (self.players_last_move[1] == (-1,-1) and self.players_last_move[2] == (-1,-1)):
            scores = self.calculate_scores()
            if scores[1] > scores[2]: 
                self.winner = 1
            else:
                self.winner = 2 
        
    def hasFinished(self):
        return self.winner != 0
    
    def PossibleMoves(self):
        moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.ValidMove(i,j):
                    moves.append((i,j))
        moves.append((-1,-1))
        return moves
    
    def MoveToAction(self, move, fill_size=0):
        if move == (-1,-1):
            return self.size**2
        i, j = move
        return j + i*self.size

    def calculate_scores(self):
        self_copy = self.copy()
        board = self_copy.board.copy() # copy the board

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
                board = self_copy.board.copy() # copy the board
                for stone in opp_group.stones:
                    board = flood_fill(stone, 3-player)
                for group in self_copy.players_groups[player]:
                    # check if the player group liberties are inside the opponent territory
                    if all(map(lambda lib_pt: (board[lib_pt]==10+(3-player) or board[lib_pt]==(3-player)), group.liberty_points)):
                        group.possible_prisioner = True
        
        # remove prisioners
        board = self_copy.board.copy() # copy the board
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
        board = self_copy.board.copy() # copy the board
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
    
    
    def rotate90(self, times):
        return np.rot90(self.board, times)

    def flip_vertical(self):
        return np.flip(self.board, 0)

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
