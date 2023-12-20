import socket
import time
import graphics
from ataxx import AttaxxBoard
import pygame

Game="A4x4" # "A6x6" "G7x7" "G9x9" "A5x5"

def parse_coords(data):
    data = data.split(sep=" ")
    coords_list = [] # go: [i, j] | attax: [i1, j1, i2, j2]
    for coord in data[1:]:
        coords_list.append(int(coord[0]))
        coords_list.append(int(coord[-1]))
    return coords_list

def start_server(host='localhost', port=12345, render=False):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(2)

    print("Waiting for two agents to connect...")
    agent1, addr1 = server_socket.accept()
    print("Agent 1 connected from", addr1)
    bs=b'AG1 '+Game.encode()
    agent1.sendall(bs)

    agent2, addr2 = server_socket.accept()
    print("Agent 2 connected from", addr2)
    bs=b'AG2 '+Game.encode()
    agent2.sendall(bs)    

    agents = [agent1, agent2]
    current_agent = 0
    
    board = AttaxxBoard(4)
    board.Start()
    play_attempts = 0
    time_to_play = time.time()

    if render:
        graphics.draw_board(board.board)
    while not board.hasFinished():
        print(board.board)
        # check if time limit to play has passed
        if time.time() - time_to_play >= 60: # more than 60 seconds to play
            agents[current_agent].sendall(b'TIME LIMIT PASSED')
            board.NextPlayer()
            current_agent = 1-current_agent
            time_to_play = time.time()
    
        try:
            data = agents[current_agent].recv(1024).decode()
            if not data:
                break

            # Process the move (example, attax: "MOVE I1,J1 I2,J2")
            print(current_agent, " -> ",data)
            i1, j1, i2, j2 = parse_coords(data)

            if board.ValidMove(i1, j1, i2, j2):
                agents[current_agent].sendall(b'VALID')
                agents[1-current_agent].sendall(data.encode())
                # show selectect piece
                if render:
                    graphics.set_selected_piece(i1,j1)
                    graphics.draw_board(board.board)
                    pygame.display.flip()
                    time.sleep(1)
                # process game
                board.Move([i1,j1,i2,j2])
                board.NextPlayer()
                # show play
                if render:
                    graphics.unselect_piece()
                    board.ShowBoard()
                    graphics.draw_board(board.board)
                    pygame.display.flip()
                # check for end of the game and switch turns
                board.CheckFinish()
                current_agent = 1-current_agent
            # invalid move
            else:
                agents[current_agent].sendall(b'INVALID')
                play_attempts += 1
                if play_attempts >= 3:
                    board.NextPlayer()
                    current_agent = 1-current_agent
            
            time.sleep(1)

        except Exception as e:
            print("Error:", e)
            break
    
    print(board.board)


    print("\n-----------------\nGAME END\n-----------------\n")
    time.sleep(1)
    agent1.close()
    agent2.close()
    server_socket.close()

def is_valid_move(move):
    # Implement the logic to check if the move is valid
    return True

if __name__ == "__main__":
    start_server(render=True)
