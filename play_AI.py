import numpy as np
import pygame
from time import sleep

import torch
torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")

import graphics
from ataxx import AtaxxBoard
from go import GoBoard
from CNN import Net
from aMCTS_parallel import MCTSParallel, MCTS_Node
from model_params import MODEL_PARAMS

GAME = "A4" # A4 - A5 - A6 - G7 - G9

def load_model(game_type="A", game_size=4, model_load_path=None):
    model = Net(**MODEL_PARAMS.get(game_type+str(game_size)))
    if model_load_path != "":
        model.load_state_dict(torch.load(model_load_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    return model

def play_game(game_type="A", game_size=4, human_player=1, model_load_path="", mcts_iterations=200, render=False):
    board = AtaxxBoard(game_size) if game_type=="A" else GoBoard(game_size)
    board.Start(render)

    if render:
        graphics.draw_board(board.board)
        pygame.display.flip()

    model = load_model(game_type, game_size, model_load_path)
    mcts = MCTSParallel(model)

    while True:
        if render:
            graphics.draw_board(board.board)
            pygame.display.flip()
        print(board.board)
        print(board.minimax(2,True,-np.inf,np.inf,board.player))

        if board.player == human_player:
            move = []
            if not render:
                move = list(map(int, input("Move: ").split(" ")))
            elif game_type == "A":
                selected = False
                while not selected:
                    click1 = graphics.piece_index_click()
                    while board.board[click1[0]][click1[1]] != board.player:
                        print("Invalid Position")
                        click1 = graphics.piece_index_click()
                    graphics.set_selected_piece(*click1)
                    move = move + list(click1)
                    graphics.draw_board(board.board)
                    pygame.display.flip()
                    click2 = graphics.piece_index_click()
                    move = move + list(click2)
                    if not (board.ValidMove(*move)):
                        print("Invalid Move")
                        graphics.unselect_piece()
                        graphics.draw_board(board.board)
                        pygame.display.flip()
                        move = []
                    else:
                        selected = True
            elif game_type == "G":
                move = tuple(graphics.piece_index_click())

            if board.ValidMove(*move):
                board.Move(move)
                board.NextPlayer()
                if render:
                    graphics.unselect_piece()
                    graphics.draw_board(board.board)
                    pygame.display.flip()
                board.CheckFinish(render)

            else:
                print("NOT A VALID MOVE")

        else:
            roots = [MCTS_Node(board)]
            mcts_probs = mcts.Search(roots, mcts_iterations, test=True)
            action = np.argmax(mcts_probs)
            move = mcts.roots[0].children[action].originMove
            print("Alphazero Move:", move)
            board.Move(move)
            board.NextPlayer()
            if render:
                graphics.draw_board(board.board)
                pygame.display.flip()
            board.CheckFinish(render)
        
        if board.hasFinished():
            print(board.board)
            if board.winner != 3:
                print(board.winner, "won")
            else:
                print("draw")
            break

if __name__ == "__main__":
    play_game("A", 4, human_player=2, model_load_path="d:/Paulo Alexandre/Ensino_Superior/3_ano/1_semestre/LabIACD/Project2/Alpha_Zero_Game_Player/Final_Models/A4_treinado_2_16.pt", mcts_iterations=1, render=True)