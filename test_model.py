from ataxx import AttaxxBoard
import torch
torch.manual_seed(0)
from CNN import Net
from aMCTS_parallel import MCTSParallel, MCTS_Node
import numpy as np

import warnings
warnings.filterwarnings("ignore")

board = AttaxxBoard(4)
board.Start()
model = Net(4,4**4,10,128)
#model.load_state_dict(torch.load("A4_0.pt", map_location="cuda" if torch.cuda.is_available() else "cpu"))
model.eval()
mcts = MCTSParallel(model)

while True:
    print(board.board)

    if board.player == 2:
        i1, j1, i2, j2 = list(map(int, input("Move: ").split(" ")))
        if board.ValidMove(i1, j1, i2, j2):
            board.Move((i1, j1, i2, j2))
            board.NextPlayer()
            board.CheckFinish()
        else:
            print("NOT A VALID MOVE")

    else:
        roots = [MCTS_Node(board)]
        mcts_probs = mcts.Search(roots, 50)

        print("\nROOT CHILDREN VISITS")
        for root_board in roots:
            for child in root_board.children.values():
                print(child.originMove, child.n)
            print()

        for i in range(len(mcts_probs)):
            if mcts.roots[0].children.get(i):
                print(mcts.roots[0].children[i].originMove, mcts_probs[i])
        action = np.argmax(mcts_probs)
        move = mcts.roots[0].children[action].originMove
        print("Alphazero Move:", move)
        board.Move(move)
        board.NextPlayer()
        board.CheckFinish()
    
    if board.hasFinished():
        print(board.board)
        if board.winner != 3:
            print(board.winner, "won")
        else:
            print("draw")
        break