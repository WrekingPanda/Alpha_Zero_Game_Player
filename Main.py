from alphazero import AlphaZero
from CNN import Net
from go import GoBoard
from ataxx import AttaxxBoard

def main():
    board_type = input()
    board_size = int(input())

    if board_type == "A" or board_type == "a":
        model = Net(board_size, board_size**4, 40, 64)
    elif board_type == "G" or board_type == "g":
        model = Net(board_size, (board_size**2)+1, 40, 64)


main()