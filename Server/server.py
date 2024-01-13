import socket
import time
import graphics
from ataxx import AtaxxBoard
from go import GoBoard
import pygame

Game="A4x4" # "A4x4" "A6x6" "G7x7" "G9x9" "A5x5"

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

    game_type = Game[0]
    game_size = int(Game[-1])
    
    board = AtaxxBoard(game_size) if game_type == "A" else GoBoard(game_size)
    board.Start()
    play_attempts = 0

    if render:
        graphics.draw_board(board.board)

    while not board.hasFinished():    
        try:
            data = agents[current_agent].recv(1024).decode()
            if not data:
                break

            # Process the move (example, attax: "MOVE i1,j1 i2,j2", go: "MOVE i, j")
            print(current_agent, " -> ", data)
            
            # check if received move indicates that the player ran out of time
            if data[-1] == "X":
                print(f"Skipping Agent {current_agent} play because the time limit was reached")
                board.NextPlayer()
                # switch agents
                current_agent = 1-current_agent
                continue

            coords = parse_coords(data)
            print(coords)
            if board.ValidMove(*coords):
                play_attempts = 0
                agents[current_agent].sendall(b'VALID')
                agents[1-current_agent].sendall(data.encode())
                # show selectect piece
                if render and game_type == "A":
                    graphics.set_selected_piece(coords[0], coords[1])
                    graphics.draw_board(board.board)
                    pygame.display.flip()
                    time.sleep(1)
                # process game
                board.Move(coords)
                board.NextPlayer()
                # show play
                if render:
                    graphics.unselect_piece()
                    graphics.draw_board(board.board)
                    pygame.display.flip()
                # check for end of the game in an Attaxx game
                if game_type == "A":
                    board.CheckFinish()
                # switch agents
                current_agent = 1-current_agent

            # invalid move
            else:
                agents[current_agent].sendall(b'INVALID')
                play_attempts += 1
                if play_attempts >= 3:
                    board.NextPlayer()
                    current_agent = 1-current_agent
            
            # might want to remove
            time.sleep(1)

        except Exception as e:
            print("Error:", e)
            break

    print("\n-----------------\nGAME END\n-----------------\n")
    winner_text = f"The winner is Agent {board.winner}!" if board.winner != 3 else "The game ended in a draw!"
    agents[current_agent].sendall(winner_text.encode())
    agents[1-current_agent].sendall(winner_text.encode())
    time.sleep(0.5)
    agent1.close()
    agent2.close()
    server_socket.close()


if __name__ == "__main__":
    start_server(render=False)
