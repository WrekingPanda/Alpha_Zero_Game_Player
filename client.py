import socket
import random
import time
import torch
import numpy as np
from ataxx import AtaxxBoard
from go import GoBoard
from CNN import Net
from aMCTS_parallel import MCTSParallel,MCTS_Node
from model_params import MODEL_PARAMS

#Game="A4x4" # "A6x6" "G7x7" "G9x9" "A5x5"

def parse_coords(data, game_type):
    # attaxx: "MOVE i,j, i,j"
    # go:     "MOVE i,j" or "PASS"
    if game_type == "A":
        coords = (int(data[5]), int(data[7]), int(data[10]), int(data[12]))
        return coords
    elif game_type == "G":
        if "P" in data: return (-1,-1)
        coords = (int(data[5]), int(data[7]))
        return coords
    return None

def load_model(game_type,game_size):
    model = Net(**MODEL_PARAMS.get(game_type+str(game_size)))
    model_name = game_type + str(game_size) + ".pt"
    model.load_state_dict(torch.load(model_name, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()

    return model

def connect_to_server(host='localhost', port=12345):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    
    response = client_socket.recv(1024).decode()
    print(f"Server ResponseINIT: {response}")
    
    Game = response[-4:]
    print("Playing:", Game)
    game_type = Game[0]
    game_size = int(Game[-1])
    board = AtaxxBoard(game_size) if game_type == "A" else GoBoard(game_size)
    board.Start(render=False)

    ag = 1 if "1" in response else 2

    current_agent = 1
    """
    time_to_play = time.time()
    """
    play_attempts = 0
    """ 
    reset_time = False
    """
    # the model is loaded here
    model = load_model(game_type,game_size)
    
    mcts = MCTSParallel(model)
    
    while True:
        if ag == current_agent:

            # get the move to execute
            roots = [MCTS_Node(board)]
            if game_type == "A":
                if game_size == 4: mcts_iters = 600   # A4
                elif game_size == 5: mcts_iters = 350 # A5
                else: mcts_iters = 115                # A6
            else:
                if game_size == 7: mcts_iters = 35    # G7
                else: mcts_iters = 7                  # G9
            mcts_probs = mcts.Search(roots, mcts_iters, test=True)

            for root_board in roots:
                for child in root_board.children.values():
                    print(child.originMove, child.n, child.p)
                print()

            for i in range(len(mcts_probs)):
                if mcts.roots[0].children.get(i):
                    print(mcts.roots[0].children[i].originMove, mcts_probs[i])
            action = np.argmax(mcts_probs)
            move = mcts.roots[0].children[action].originMove
            """ purelly for testing, we can delete the following line later """
            print("Alphazero Move:", move)

            if game_type == "A":
                move_msg = "MOVE " + str(move[0]) + "," + str(move[1]) + " " + str(move[2]) + "," + str(move[3])
            elif move != (-1,-1):
                move_msg = "MOVE " + str(move[0]) + "," + str(move[1])
            else:
                move_msg = "PASS"
            msg = str(move_msg)
            try:
                client_socket.send(msg.encode())
                print("Send:", msg)
            except Exception as e:
                print(f"Agent {ag} Error: {e}")
                client_socket.close()
                break
        
        # Wait for server response
        response = client_socket.recv(1024).decode()
        print(f"Server Response1: {response}")
        if "END" in response: break
        elif "INVALID" in response:
            play_attempts += 1
            # repeats the cycle without changing the current player
            if play_attempts < 3: continue
            # switches turns
            else:
                play_attempts = 0
                board.NextPlayer()
                current_agent = 3-current_agent
        elif "VALID" in response:
            coords = parse_coords(msg, game_type)
            # process game
            board.Move(coords)
            board.NextPlayer()
            current_agent = 3-current_agent
        elif "MOVE" in response or "PASS" in response:
            coords = parse_coords(response, game_type)
            # process game
            board.Move(coords)
            board.NextPlayer()
            current_agent = 3-current_agent
        
        print("-------------------------")
        print(board.board)
        print("-------------------------")

if __name__ == "__main__":
    connect_to_server()
