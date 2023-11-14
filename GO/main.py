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
        row, col = int(input_str[0]), int(input_str[1])

        if(board.is_valid_move(row, col)):
            board.PutPiece(row, col)

            #continue ... 

        else:
            print("Invalid Move!")
        