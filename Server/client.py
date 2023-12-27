import socket
import random
import time
from ataxx import AttaxxBoard
from go import GoBoard

#Game="A4x4" # "A6x6" "G7x7" "G9x9" "A5x5"

def generate_random_move():
    x = random.randint(0, 9)
    y = random.randint(0, 9)
    return f"MOVE {x},{y}"

def generate_random_move2():
    x = random.randint(0, 4)
    y = random.randint(0, 4)
    x2 = random.randint(0, 4)
    y2 = random.randint(0, 4)
    return f"MOVE {x},{y},{x2},{y2}"

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
    # model = ...
    
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
            
            # get move to perform
            # this is where the predicition of the model should be interegrated
            # the model should expect the input to be a bidimensional numpy array
            # in this implementation, both structures holding the configuration of the Ataxx and Go boards are done using numpy arrays
            # the model's output will have to be formatted such that if follows the structure "MOVE row_index,column_index" (Go games) or "MOVE from_row_index,from_column_index to_row_index,to_column_index" (Ataxx games)
            # it is strongly suggested to warn the students that a method .get_move_to_play(input: board configuration in numpy array) should be implemented in their models that performs the necessary formatation to text the server
            # move = model.get_move_to_play(input=board.board)
            
            # as a placeholder, the server will receive the move to be performed directly through the standard input
            move = input()
            client_socket.send(move.encode())
            print("Send:", move)
        
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
            coords = parse_coords(move)
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
