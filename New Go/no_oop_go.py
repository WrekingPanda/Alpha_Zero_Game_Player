import numpy as np
from pprint import pprint

MARKER = 4

class GoFunctional:
    liberties = []
    block = []

    @staticmethod
    def is_in(i, j, board_size):
        return 0 <= i < board_size and 0 <= j < board_size
    
    @staticmethod
    def NextPlayer(player):
        return 3-player
    
    @staticmethod
    def ValidMove(i, j, board: np.ndarray, player: int, last_two_boards):
        coords = (i,j)
        if coords == (-1,-1):
            return True
        if not GoFunctional.is_in(i, j, board.shape[0]): return False
        if board[coords]: return False
        board_copy = np.copy(board)
        last_two_boards_copy = np.copy(last_two_boards)
        GoFunctional.Move(board_copy, coords, player, last_two_boards_copy)
        return board_copy.tolist() not in last_two_boards

    @staticmethod
    def Move(board, coords, player, last_two_boards):
        last_two_boards[0] = last_two_boards[1]
        last_two_boards[2] = last_two_boards[2]
        if coords != (-1,-1):
            board[coords] = player
            # handle possible captures
            GoFunctional.captures(board, 3-player)
        # update last board
        last_two_boards[2] = board.tolist()

    @staticmethod
    def captures(board, opp_player):
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i,j] == opp_player:
                    GoFunctional.count(board, (i,j), opp_player)
                    if len(GoFunctional.liberties) == 0:
                        GoFunctional.clear_block(board)
                    GoFunctional.restore_board(board)

    @staticmethod
    def count(board, coords, player):
        piece = board[coords]
        if piece == player and piece < MARKER:
            GoFunctional.block.append(coords)
            board[coords] += MARKER
            i, j = coords
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                if GoFunctional.is_in(i+di, j+dj, board.shape[0]):
                    GoFunctional.count(board, (i+di, j+dj), player)
        elif piece == 0:
            GoFunctional.liberties.append(coords)
    
    @staticmethod
    def clear_block(board):
        for coords in GoFunctional.block:
            board[coords] = 0

    @staticmethod
    def restore_board(board):
        GoFunctional.clear_groups()
        # unmark stones
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i,j] > MARKER:
                    board[i,j] -= MARKER

    @staticmethod
    def clear_groups():
        GoFunctional.block = []
        GoFunctional.liberties = []

    @staticmethod
    def CheckFinish(board, player, last_two_boards):
        if len(last_two_boards) < 3: return 0 # there must have been played at least 2 moves
        if last_two_boards[1] != last_two_boards[2]: return 0 # not over yet
        # normal scenario
        scores = GoFunctional.calculate_scores(board, player)
        if scores[1] > scores[2]:
            return 1 # player 1 is the winner
        return 2     # player 2 is the winner

    @staticmethod
    def calculate_scores(board:np.ndarray, cur_player):
        possible_prisioners = []

        def flood_fill(board:np.ndarray, coords, player):
            if board[coords] == 0 or (board[coords] != 1 and board[coords] != 2): board[coords] += 10+player
            i,j = coords
            for (di, dj) in [(-1,0),(1,0),(0,-1),(0,1)]:
                if GoFunctional.is_in(i+di, j+dj, board.shape[0]) and board[i+di, j+dj] != 1 and board[i+di, j+dj] != 2 and board[i+di, j+dj] != 10+player and board[i+di, j+dj] < 15:
                    flood_fill(board, (i+di, j+dj), player)
            return board
        
        def check_possible_prisioners(board:np.ndarray, player):
            nonlocal possible_prisioners
            for i in range(board.shape[0]):
                for j in range(board.shape[1]):
                    if board[i,j] == player:
                        GoFunctional.count(board, (i,j), player)
                        if len(GoFunctional.liberties) == 0:
                            if GoFunctional.block[0] not in possible_prisioners:
                                possible_prisioners += GoFunctional.block
                        GoFunctional.restore_board(board)

        # check for possible prisioners
        board_copy = np.copy(board)
        for player in [cur_player, 3-cur_player]:
            for i in range(board.shape[0]):
                for j in range(board.shape[1]):
                    if board_copy[i,j] == 3-player:
                        board_copy = np.copy(board)
                        board_copy = flood_fill(board_copy, (i,j), 3-player)
            check_possible_prisioners(board_copy, player)
        
        # eliminate possible prisioners
        board_copy = np.copy(board)
        for coords in possible_prisioners:
            board_copy[coords] = 0
        
        # calculate territories
        for player in [cur_player, 3-cur_player]:
            for i in range(board.shape[0]):
                for j in range(board.shape[1]):
                    if board_copy[i,j] == player:
                        board_copy = flood_fill(board_copy, (i,j), player)

        # scores
        scores = {1:0, 2:0}
        # count territoy
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board_copy[i,j] == 11: scores[1] += 1
                if board_copy[i,j] == 12: scores[2] += 1
        # komi
        scores[2] += 5.5
        return scores
    
    @staticmethod
    def PossibleMoves(board, cur_player, last_two_boards):
        moves = []
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if GoFunctional.ValidMove(i, j, board, cur_player, last_two_boards):
                    moves.append((i,j))
        moves.append((-1,-1))
        return moves

    @staticmethod
    def MoveToAction(move, board_size, fill_size=0):
        if move == (-1,-1):
            return board_size**2
        i, j = move
        return j + i*board_size

def rotate90(board, times):
    return np.rot90(np.copy(board), times)

def flip_vertical(board):
    return np.flip(np.copy(board), 0)

def EncodedGameStateChanged(board, cur_player, fill_size=0):
    return np.stack(
        (board == cur_player, board == 3-cur_player, board == 0)
    ).astype(np.float32)