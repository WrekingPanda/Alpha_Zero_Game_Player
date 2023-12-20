import socket
import random
import time

Game="A4x4" # "A6x6" "G7x7" "G9x9" "A5x5"

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

def connect_to_server(host='localhost', port=12345):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    
    response = client_socket.recv(1024).decode()
    print(f"Server ResponseINIT: {response}")
    
    Game = response[-4:]
    print("Playing:", Game)
    
    if "1" in response:
        ag=1
    else:
        ag=2
    first=True
    
    while True:
        # Generate and send a random move
        
        if ag == 1 or not first:
            move = input()
            time.sleep(1)
            client_socket.send(move.encode())
            print("Send:",move)
        
            # Wait for server response
            response = client_socket.recv(1024).decode()
            print(f"Server Response1: {response}")
            if "END" in response: break
         
        first=False
        response = client_socket.recv(1024).decode()
        print(f"Server Response2: {response}")
        if "END" in response: break

        # Add some condition to break the loop, if necessary
        # Example: If server sends a certain message, or after a number of moves

    client_socket.close()

if __name__ == "__main__":
    connect_to_server()
