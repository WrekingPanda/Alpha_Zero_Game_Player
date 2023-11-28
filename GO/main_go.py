from board_go import board_go

board = board_go(7)

while True:
    Pass = False

    board.Print()
    player = board.player
    if player == 1:
        p = "Black"
    else: 
        p = "White"
    print("It's " + p + "'s turn!")
    print("Waiting move...")

    input_str = input()
    if(input_str == 'p'):
        board.PassTurn()

    else:
        list_string = input_str.split(' ')
        list_int = [int(x) for x in list_string[:2]]       
        row, col = list_int[0], list_int[1]
        
        if(board.is_valid_move(row, col)):
            board.PutPiece(row, col)

                #continue ... 

        else:
            print("Invalid Move! PLease try again.")
            