import numpy as np
from group import Group


BLACK = True
WHITE = False
EMPTY = None

KOMI = 5.5

class Board_go:

    def __init__(self, size):
        # Game attributes
        self.size = size
        self.turn = BLACK

        # Used for Ko-rule
        self.blocked_field = None

        # Used to detect if players pass
        self.has_passed = False

        # Game over flag
        self.game_over = False

        # The board is represented by a matrix size*size
        self.board = [[EMPTY for i in range(self.size)] for j in range(self.size)]   
        self.territory = [[EMPTY for i in range(self.size)] for j in range(self.size)]
        
        # Score from empty fields at the end of the game
        self.score = [0,0]

        # Stones killed during game
        self.captured = [0,0]
    


    #########################################################################################
    # PRINT #
    #########################################################################################
    def get_matrix(self):
        matrix = np.zeros((self.size,self.size))

        for i in range(self.size):
            for j in range(self.size):
                if(self.board[i][j] is None):
                    matrix[i][j] = 0
                # Black pieces
                elif(self.board[i][j].color):
                    matrix[i][j] = 1
                # White pieces
                else:
                    matrix[i][j] = 2
        
        return matrix


    def _print(self):
        m = self.get_matrix()

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
                if(m[i][j] == 0): 
                    row += " . "
                    
                elif(m[i][j] == 1): 
                    row += " X "

                elif(m[i][j] == 2): 
                    row += " O "
                
            row += "|"    
            print(row)

        print(line)


    #########################################################################################
    # GAME MECHANICS #
    #########################################################################################

    # Action when a player passes his turn 
    def passing(self):
        # Do  nothing if game is over
        if self.game_over:
            return False

        # Both players have passed = game over
        if self.has_passed:
            self.game_over = True
            return True
    
        # invert turns and set passed to true
        self.turn = WHITE if (self.turn==BLACK) else BLACK
        self.has_passed = True
        self.blocked_field = None

        return True


    # Returns a list containign the colors of each stone in the board 
    def _stones(self):
        # Initialize list 
        colors = [[None for i in range(self.size)] for j in range(self.size)]

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] is None:
                    colors[i][j] = 0
                elif self.board[i][j].color:
                    colors[i][j] = 1
                else:
                    colors[i][j] = 2

        return colors
    

    # Add a group of stones to the game
    def _add(self, group):
        for (x, y) in group.stones:
            self.board[x][y] = group


    # Removes a group of stones from the game
    def _remove(self, group):
        for (x, y) in group.stones:
            self.board[x][y] = None


    # Removes a group of stones from the game a increases the counter capture
    def _kill(self, group):
        self.captured[not group.color] += group.size
        self._remove(group)


    # Counts number of liberties from a group
    def _liberties(self, group):
        return sum(1 for u, v in group.border if self.board[v][u] is None)

    def update_liberties(self, added_stone=None):
    
        '''
        Updates the liberties of the entire board, group by group.
        Usually a stone is added each turn. To allow killing by 'suicide',
        all the 'old' groups should be updated before the newly added one.
        '''
        
        for group in self.groups:
            if added_stone:
                if group == added_stone.group:
                    continue
            group.update_liberties()
        if added_stone:
            added_stone.group.update_liberties()


    # Sum up the scores = empty fields + captured stones per player
    def add_scores(self):
        return [self.score[0] + self.captured[0], self.score[1] + self.captured[1]]    


    # Returns useful data 
    def get_data(self):
        data = {
            'size'      : self.size,
            'stones'    : self._stones(),
            'territory' : self.territory,
            'game_over' : self.game_over,
            'score'     : self.add_scores(),
            'color'     : self.turn
        }

        return data


    # Attempts to place a stone
    def place_stone(self, x, y):
        if self.game_over:
            return False
        
        if self.board[x][y] is not None:
            return False
        
        if self.blocked_field == (x, y):
            return False
        

        new_group = Group(stones=[(x,y)], color=self.turn)
        group_to_remove = []
        group_to_kill = []


        ############################
        # Move Validation
        ############################

        is_valid = False

        # All direct neighboors of (x,y)
        for (u, v) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if(u<0 or v<0 or u>=self.size or v>=self.size):
                continue

            # Add neighbor to the border of the new group
            new_group.border.add((u,v))
            other_group = self.board[u][v]

            # Check if neighbor is None
            if other_group is None:
                is_valid = True
                continue

            # Same color
            if(new_group.color == other_group.color):
                # Merge 2 groups
                new_group = new_group + other_group

                group_to_remove.append(other_group)

            # Different color groups
            # Check is there's only one liberty left 
            elif self._liberties(other_group) == 1:
                is_valid = True

                if other_group not in group_to_kill:
                    group_to_kill.append(other_group)

        # New group has at least 1 liberty 
        if self._liberties(new_group) >= 1:
            is_valid = True

    
        ############################
        # Move Exectution
        ############################
        
        # Move is valid
        if is_valid:
            # Remove groups
            for grp in group_to_remove:
                self._remove(grp)
            
            # Kill groups 
            for grp in group_to_kill:
                self._kill(grp)

            # Add new group 
            self._add(new_group)

        # Move is invalid
        else:
            return False
        

        ######################################
        # ko-rule
        ######################################
        # 3 conditions for the ko-rule to apply
        # 1. the new group has only one stone
        # 2. only one group has been killed
        # 3. the killed group has only had one stone

        if new_group.size == 1 and len(group_to_kill) == 1 and group_to_kill[0].size == 1:
            for (x, y) in group_to_kill[0].stones:
                self.blocked_field = (x, y)
            
        else:
            self.blocked_field = None
        

        ######################################
        # Turn End Actions: Change the current player
        ######################################
        # Switch player
        self.turn = WHITE if (self.turn == BLACK) else BLACK
        self.has_passed == False

        return True


    # Counts number of marked fields and update score
    def _compute_score(self):
        # Reset the scores to zero
        self.score = [0,0]

        for i in range(self.size):
            for j in range(self.size):

                # Count black stones
                if self.territory[i][j] == BLACK:
                    if self.board[i][j] != None:
                        # add 1 point for each stone inside territory
                        self.score[BLACK] += 2
                    else:
                        self.score[BLACK] += 1
                
                # Count white stones
                elif self.territory[i][j] == WHITE:
                    if self.board[i][j] != None:
                        # add 1 point for each stone inside territory
                        self.score[WHITE] += 2
                    else:
                        self.score[WHITE] += 1
    

    # Claims an empty field including all adjacent empty fields 
    def _claim_empty(self, x, y, color, area=None):
        if area is None:
            area = list()
        
        if self.board[x][y] is not None or (x,y) in area:
            return
    
        # Claim empty field
        self.territory[x][y] = color

        # Remember the current location
        area.append((x,y))

        # Check all neighbor recursivly 
        for (u, v) in [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]:

            # Check that the neighbor actually exists on the board
            if u < 0 or v < 0 or u >= self.size or v >= self.size:
                continue
            
            # Claim neighboring empty field
            if (u, v) not in area and self.board[v][u] is None:
                self._claim_empty(u, v, color, area=area)


    # Claims an entire group and also all adjacent empty fields
    def _claim_group(self, x, y, color, area=None):
        if area is None:
            area = list()

        # Claming each stone in the group at coordinates(x,y)
        for (u, v) in self.board[x][y].stones:
            if (u,v) not in area:
                # Remember current location
                area.append((u,v))

                # Claim position
                self.territory[u][v] = color

        # Claim each empty firlds in the adjacent empty fields
        for (u, v) in self.board[y][x].border:
            if self.board[v][u] is None and (u, v) not in area:
                self._claim_empty(u, v, color, area=area)  
        

    # Finds the connected empty fields starting at (x,y) and counts the adjacent stones
    def _find_empty(self, x, y, area=None, count=None):
        if area is None:
            area = list()

        if count is None:
            count = [0,0]

        if self.board[x][y] is not None or (x,y) in area:
            return area, count

        for (u, v) in [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]:
            if u < 0 or v < 0 or u >= self.size or v >= self.size:
                continue
            
            # claim neighboring empty field
            if (u, v) not in area:
                if self.board[v][u] is None:
                    self._find_empty(u, v, area=area, count=count)
                else:
                    count[self.board[v][u].color] += 1


        return area, count
    

    '''Function that can be evoked by user to claim territory for one player.
        For empty fields it will also mark all adjacent empty fields,
        for fields that contain a stone it will mark the entire stone
        group and all adjacent empty spaces.'''
    def mark_territory(self, x, y):
        if not self.game_over:
            return
        
        # Claim an empry field
        if self.board[x][y] in None:
            # cycle through the colours depending on how the field is currently marked
            # None => Black => White => None
            col_dict = {None:BLACK, BLACK:WHITE, WHITE:None}

            color = col_dict[self.territory[x][y]]
            self._claim_empty(x, y, color)

        # Claim a group 
        else:
            # Choose whether to mark or unmark a group 
            if self.territory[x][y] is None:
                color = not self.board[x][y].color
            else:
                color = None

            # Recursively claim the fields
            self._claim_empty(x, y, color)

        # Compute the score 
        self._compute_score()

    
    # Tries to autmatically claim territory for the proper players
    def find_territory(self):
        # Keep track of th checked fields
        covered_area = list()

        for x in range(self.size):
            for y in range(self.size):
                if (x,y) in covered_area:
                    continue

                if self.board[x][y] is None:
                    # Find all adjacent empty fields
                    # Count contains the number of adjacent stones of each color.
                    area, count = self._find_empty(x, y, area=covered_area)
                    covered_area += area

                    # Claim the territory if black has no adjacent stones
                    if count[BLACK] == 0 and count[WHITE] > 0:
                        self._claim_empty(x, y, WHITE)

                    # Claim the territory if white has no adjacent stones
                    elif count[WHITE] == 0 and count[BLACK] > 0:
                        self._claim_empty(x, y, BLACK)

        # Compute the score 
                        self._compute_score()




'''    #########################################################################################
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
    def count_liberties(self, row, col, visited=None, player = 1):

        if visited is None:
            visited = set()
            player = self.board[row][col]

        # Check if the current position is out of bounds or already visited
        if not (0 <= row < self.size and 0 <= col < self.size) or (row, col) in visited:
            return 0
        

        visited.add((row, col))

        # Check if the current position is empty, in which case it contributes a liberty
        if self.board[row][col] == 0:
            return 1

        # Check if the current position has a piece of the opponent player, in wich case it stops counting
        if self.board[row][col] != player:
            return 0

        # Check the neighboring positions recursively
        liberties = 0
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row = row + dr
            new_col = col + dc
            liberties += self.count_liberties(new_row, new_col, visited,player)
        return liberties

    def count_group_liberties(self, row, col):
        stone_color = self.board[row][col]
        if stone_color == EMPTY:
            return 0  # Empty intersection has no liberties

        visited = set()
        return self.count_liberties(row, col)
    

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
'''