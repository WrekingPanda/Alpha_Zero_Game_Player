import socket
import random
import time
import torch
import numpy as np
from ataxx import AttaxxBoard
from fastgo import GoBoard
from CNN import Net
from aMCTS_parallel import MCTSParallel,MCTS_Node
from model_params import MODEL_PARAMS

#Game="A4x4" # "A6x6" "G7x7" "G9x9" "A5x5"

def parse_coords(data):
    data = data.split(sep=" ")
    coords_list = [] # go: [i, j] | attax: [i1, j1, i2, j2]
    for coord in data[1:]:
        if "-" in coord:
            coords_list.append(-1)
            coords_list.append(-1)
        else:
            coords_list.append(int(coord[0]))
            coords_list.append(int(coord[-1]))
    coords_list = tuple(coords_list)
    return coords_list

def load_model(game_type,game_size):
    model = Net(**MODEL_PARAMS.get(game_type+str(game_size)))
    
    model_path = "D:/Paulo Alexandre/Ensino_Superior/3_ano/1_semestre/LabIACD/Project2/Alpha_Zero_Game_Player/"
    model_name = game_type + "_" + str(game_size) + ".pt"
    
    model.load_state_dict(torch.load(model_path + model_name, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
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
    board = AttaxxBoard(game_size) if game_type == "A" else GoBoard(game_size)
    board.Start(render=False)

    ag = 1 if "1" in response else 2

    current_agent = 1
    time_to_play = time.time()
    play_attempts = 0
    reset_time = False

    # the model is loaded here
    model = load_model(game_type,game_size)
    
    mcts = MCTSParallel(model)
    
    while True:
        if ag == current_agent:
            if reset_time:
                time_to_play = time.time()
                reset_time = False
            # check if time limit to play has passed
            if time.time() - time_to_play >= 600: # more than 60 seconds to play
                # message to indicate no move was decided in the set playtime
                move = "MOVE X,X X,X" if game_type == "A" else "MOVE X,X"
                client_socket.send(move.encode())
                board.NextPlayer()
                current_agent = 3-current_agent
                reset_time = True
                continue
        
            # get the move to execute
            roots = [MCTS_Node(board)]
            mcts_probs = mcts.Search(roots, 200, test=True)

            for root_board in roots:
                for child in root_board.children.values():
                    print(child.originMove, child.n, child.p)
                print()

            for i in range(len(mcts_probs)):
                if mcts.roots[0].children.get(i):
                    print(mcts.roots[0].children[i].originMove, mcts_probs[i])
            action = np.argmax(mcts_probs)
            move = mcts.roots[0].children[action].originMove
            # move = (i1, j1, i2, j2)
            # mive_msg = "MOVE i1,j1 i2,j2"
            """ purelly for testing, we can delete the following line later """
            print("Alphazero Move:", move)

            if game_type == "A":
                move_msg = "MOVE " + str(move[0]) + "," + str(move[1]) + " " + str(move[2]) + "," + str(move[3])
            else:
                move_msg = "MOVE " + str(move[0]) + "," + str(move[1])
            msg = str(move_msg)
            client_socket.send(msg.encode())
            print("Send:", msg)
        
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
            coords = parse_coords(msg)
            # process game
            board.Move(coords)
            board.NextPlayer()
            current_agent = 3-current_agent
        elif "MOVE" in response:
            coords = parse_coords(response)
            # process game
            board.Move(coords)
            board.NextPlayer()
            current_agent = 3-current_agent
        
        print("-------------------------")
        print(board.board)
        print("-------------------------")

    client_socket.close()

if __name__ == "__main__":
    connect_to_server()
