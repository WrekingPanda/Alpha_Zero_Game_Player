from board_go import Board_go

print("Please select a board size:")
print("1. 7x7")
print("2. 9x9")

size_valid = False
while not size_valid:
    size = int(input())
    if size == 1:
        size = 7
        size_valid = True
    elif size == 2:
        size = 9
        size_valid = True
    elif size == 4:
        size_valid = True
    else:
        print("Invalid size, please choose between 1 or 2.") 


board = Board_go(size=size)

while(not board.game_over):
    board._print()
    if board.turn:
        print(f"It's Black's turn!")
    else:
        print(f"It's White's turn!")

    # Read input move 
    input_str = input()

    # Pass turn 
    if(input_str == 'p'):
        board.passing()

    else:
        list_string = input_str.split(' ')
        list_int = [int(x) for x in list_string[:2]]       
        row, col = list_int[0], list_int[1]
        
        if(row<0 or col<0 or row>=size or col>=size):
            continue
        else :
            board.place_stone(row, col)
            board._compute_score()

    #print(board.get_data())
    print("Score: ")
    print(board._compute_score())
    print("Stones: ")
    print(board._stones())
    print("Liberties: ")
    #print(board._liberties(1))        
        





