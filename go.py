import numpy as np
from copy import deepcopy
import pygame
import graphics

class Group:
    def __init__(self, stone_coords, go_board):
        # Initialize group with a single stone
        self.stones = set([stone_coords])
        self.possible_prisioner = False  # Flag indicating if the group could be a potential prisoner

        # Find neighboring groups belonging to the same player
        neighbor_groups = set()
        i, j = stone_coords
        for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if go_board.is_in(i + di, j + dj) and go_board.board[i + di, j + dj] == go_board.player:
                neighbor_groups.add(go_board.group_of_stone[(i + di, j + dj)])

        # Merge with neighboring groups and update the game state
        for neighbor_group in neighbor_groups:
            self.stones = self.stones.union(neighbor_group.stones)
            go_board.players_groups[go_board.player].remove(neighbor_group)

        # Update group references on the board
        for stone in self.stones:
            go_board.group_of_stone[stone] = self
        go_board.board[stone_coords] = go_board.player

        # Calculate liberty points for the group
        self.liberty_points = self.get_liberty_points(go_board)
        self.liberties = len(self.liberty_points)

    # Get liberty points for the group
    def get_liberty_points(self, go_board):
        liberty_pts = set()
        for (stone_i, stone_j) in self.stones:
            for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (go_board.is_in(stone_i + di, stone_j + dj)) and (
                        go_board.board[stone_i + di, stone_j + dj] == 0):
                    liberty_pts.add((stone_i + di, stone_j + dj))
        return liberty_pts

    # Count the number of liberties for the group
    def count_liberties(self):
        return len(list(self.liberty_points))

    # Override equality check for comparing groups
    def __eq__(self, other):
        return self.stones.__eq__(other.stones)

    # Override hash function for group objects
    def __hash__(self):
        return object.__hash__(self)


# Class representing the Go board
class GoBoard:
    def __init__(self, dim):
        self.size = dim
        self.board = np.zeros(shape=(dim, dim), dtype=np.float32)
        self.player = 1
        self.players_groups = {1: [], 2: []}  # Dictionary holding player groups
        self.group_of_stone = {(i, j): None for j in range(dim) for i in range(dim)}
        self.players_prev_boards = {1: None, 2: None}  # Store previous board states for repetition check
        self.players_captured_opp_stones = {1: 0, 2: 0}  # Count of opponent stones captured by each player
        self.players_last_move = {1: (-2, -2), 2: (-2, -2)}  # Last move made by each player
        self.winner = 0  # Winner of the game

    # Create a deep copy of the current board
    def copy(self):
        return deepcopy(self)

    # Override equality check for comparing board states
    def __eq__(self, other):
        return self.board.tolist() == other.board.tolist()

    # String representation of the board for display purposes
    def __str__(self):
        out = ""
        for _ in range(2 * self.size + 1): out += "-"
        out += "\n"
        for i in range(self.size):
            for j in range(-1, self.size + 1):
                if j == -1 or j == self.size:
                    out += "|"
                elif j == 0:
                    out += str(self.board[i, j])
                else:
                    out += " " + str(self.board[i, j])
            out += "\n"
        for _ in range(2 * self.size + 1): out += "-"
        return out

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
        coords = (i, j)

        # Pass play
        if coords == (-1, -1):
            return True

        if self.is_in(i, j):
            # Non-empty cell
            if self.board[coords] != 0:
                return False

            # Repetition of board
            if self.players_last_move[self.player] == coords:
                return False

            # Create a copy of the current board and simulate the move
            copy = deepcopy(self)
            Group(coords, copy)
            copy.players_groups[copy.player].append(copy.group_of_stone[coords])
            copy.update_player_groups(coords)

            # Check if the new group has liberties
            return (
                    (self.players_prev_boards[self.player] == None) or
                    (self.players_prev_boards[self.player] != None and self.players_prev_boards[self.player] != str(
                        copy.board))
            ) and copy.group_of_stone[coords].count_liberties() != 0

        # Out of bounds
        return False

    # Update player groups after a move
    def update_player_groups(self, move):
        if move == (-1, -1):
            return

        played_stone = move
        adj_stones = [(played_stone[0] + di, played_stone[1] + dj) for di, dj in
                      [(-1, 0), (1, 0), (0, -1), (0, 1)] if self.is_in(played_stone[0] + di, played_stone[1] + dj)]
        updated_groups = set()

        for adj_stone in adj_stones:
            # Stone of the same group
            if self.board[adj_stone] == self.board[played_stone] and self.group_of_stone[played_stone] not in updated_groups:
                # Player group
                p_group = self.group_of_stone[played_stone]
                p_group.liberty_points = p_group.get_liberty_points(self)
                p_group.liberties = p_group.count_liberties()
                updated_groups.add(p_group)

            # Stone of the opponent
            elif self.board[adj_stone] == 3 - self.player:
                opp_group = self.group_of_stone[adj_stone]
                if opp_group not in updated_groups:
                    # Opponent group lost all liberties -> Opponent group to be deleted & player groups' liberties need to be updated
                    if len(opp_group.liberty_points) == 1 and played_stone in opp_group.liberty_points:
                        p_groups_to_be_updated = []
                        self.players_captured_opp_stones[self.player] += len(opp_group.stones)

                        for opp_stone in opp_group.stones:
                            # Look for player's groups adjacent to the opponent group to be deleted
                            opp_adj_stones = [
                                (opp_stone[0] + di, opp_stone[1] + dj) for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                if self.is_in(opp_stone[0] + di, opp_stone[1] + dj)]
                            for opp_adj_stone in opp_adj_stones:
                                if self.board[opp_adj_stone] == self.player:
                                    p_groups_to_be_updated.append(self.group_of_stone[opp_adj_stone])

                            # Delete opponent group
                            self.board[opp_stone] = 0
                            self.group_of_stone[opp_stone] = None

                        # Update registered player's groups
                        for p_group in p_groups_to_be_updated:
                            p_group.liberty_points = p_group.get_liberty_points(self)
                            p_group.liberties = p_group.count_liberties()

                    # Remove the liberty point
                    elif len(opp_group.liberty_points) > 1:
                        opp_group.liberty_points.remove(played_stone)
                        opp_group.liberties -= 1
                        updated_groups.add(opp_group)

    # Perform a move and update the board state
    def Move(self, coords):
        self.players_last_move[self.player] = coords
        if coords != (-1, -1):
            Group(coords, self)
            self.players_groups[self.player].append(self.group_of_stone[coords])
            self.update_player_groups(coords)
            self.players_prev_boards[self.player] = str(self.board)
        elif self.hasFinished():
            scores = self.calculate_scores()
            self.winner = 1 if scores[1] > scores[2] else 2

    # Check if the game has finished
    def CheckFinish(self):
        if (self.players_last_move[1] == (-1, -1) and self.players_last_move[2] == (-1, -1)):
            scores = self.calculate_scores()
            if scores[1] > scores[2]:
                self.winner = 1
            else:
                self.winner = 2

    # Check if the game has finished
    def hasFinished(self):
        return self.winner != 0

    # Generate possible moves for the current player
    def PossibleMoves(self):
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.ValidMove(i, j):
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
        self_copy = deepcopy(self)
        board = deepcopy(self_copy.board)  # Copy the board

        def flood_fill(coords, player):
            if board[coords] == 0 or (board[coords] != 1 and board[coords] != 2): board[coords] += 10 + player
            i, j = coords
            for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if self_copy.is_in(i + di, j + dj) and board[i + di, j + dj] != 1 and board[i + di, j + dj] != 2 and \
                        board[i + di, j + dj] != 10 + player and board[i + di, j + dj] < 15:
                    flood_fill((i + di, j + dj), player)
            return board

        # Check for possible prisoners
        for player in [self_copy.player, 3 - self_copy.player]:
            for opp_group in self_copy.players_groups[3 - player]:
                board = deepcopy(self_copy.board)  # Copy the board
                for stone in opp_group.stones:
                    board = flood_fill(stone, 3 - player)
                for group in self_copy.players_groups[player]:
                    # Check if the player group liberties are inside the opponent territory
                    if all(
                            map(lambda lib_pt: (board[lib_pt] == 10 + (3 - player) or board[lib_pt] == (3 - player)),
                                group.liberty_points)):
                        group.possible_prisioner = True

        # Remove prisoners
        board = deepcopy(self_copy.board)  # Copy the board
        for player in [self_copy.player, 3 - self_copy.player]:
            for opp_group in self_copy.players_groups[3 - player]:
                if opp_group.possible_prisioner:
                    continue
                for stone in opp_group.stones:
                    board = flood_fill(stone, 3 - player)
                for group in deepcopy(self_copy.players_groups[player]):
                    # Check if the player group liberties are inside the opponent territory
                    if group.possible_prisioner:
                        for stone in group.stones:
                            self_copy.board[stone] = 0
                            self_copy.group_of_stone[stone] = None
                        self_copy.players_groups[player].remove(group)

        # Calculate territories
        board = deepcopy(self_copy.board)  # Copy the board
        for player in [3 - self_copy.player, self_copy.player]:
            for group in self_copy.players_groups[player]:
                for stone in group.stones:
                    board = flood_fill(stone, player)

        # Scores
        scores = {1: 0, 2: 0}
        # Count territory
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i, j] == 11: scores[1] += 1
                if board[i, j] == 12: scores[2] += 1
        # Subtract the number of stones captured by the opponent
        scores[1] -= self_copy.players_captured_opp_stones[2]
        scores[2] -= self_copy.players_captured_opp_stones[1]
        # Komi
        scores[2] += 5.5
        return scores

    # Rotate the board 90 degrees clockwise 'times' times
    def rotate90(self, times):
        copy = deepcopy(self)
        copy.board = np.rot90(copy.board, times)
        return copy



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