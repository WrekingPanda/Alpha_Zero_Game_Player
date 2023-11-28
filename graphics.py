import pygame
import numpy as np
import sys

from typing import Literal

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
SCREEN_PADDING = 100

WHITE = (255, 255, 255)
LIGHT_GREY = (200, 200, 200)
GREY = (125, 125, 125)
DARK_GREY = (50, 50, 50)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WOOD_BROWN = (156, 109, 51)

GRID_LINES_WIDTH = 4

GAME_TITLE: Literal['attax', 'a', 'go', 'g'] = None
SCREEN: pygame.Surface = None
GRID: np.ndarray = None

BG_COLOR = None
WINDOW_TITLE = None

CELL_LENGTH = None
PIECE_RADIUS = None
PIECE_COLOR = None


def is_in_grid(i: int, j: int):
    return (0 <= i and i < GRID.shape[0]) and (0 <= j and j < GRID.shape[1])


def SET_GLOBALS(game_title: Literal['attax', 'a', 'go', 'g'], grid: np.ndarray):
    global GAME_TITLE, GRID, BG_COLOR, WINDOW_TITLE, CELL_LENGTH, PIECE_RADIUS, PIECE_COLOR
    GAME_TITLE = game_title
    GRID = grid
    BG_COLOR = LIGHT_GREY if game_title == "a" or game_title == "attax" else WOOD_BROWN
    WINDOW_TITLE =  "Attax" if game_title == "a" or game_title == "attax" else "Go"
    CELL_LENGTH = (min(SCREEN_HEIGHT, SCREEN_WIDTH) - 2*SCREEN_PADDING) // max(grid.shape[0], grid.shape[1])
    PIECE_RADIUS = CELL_LENGTH//2 - CELL_LENGTH//8
    PIECE_COLOR = {1: RED, 2: BLUE} if GAME_TITLE == "attax" or GAME_TITLE == "a" else {1: WHITE, 2: BLACK}


# sets the game window
def SET_SCREEN():
    global SCREEN
    SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(WINDOW_TITLE)
    SCREEN.fill(BG_COLOR)


def draw_piece(i: int, j: int):
    if  (0 <= i and i < GRID.shape[0]) and (0 <= j and j < GRID.shape[1]):
        piece_center_x = SCREEN_PADDING + j*CELL_LENGTH + CELL_LENGTH//2
        piece_center_y = SCREEN_PADDING + i*CELL_LENGTH + CELL_LENGTH//2
        pygame.draw.circle(
            surface=SCREEN, color=PIECE_COLOR[GRID[i][j]],
            center=(piece_center_x, piece_center_y), radius=PIECE_RADIUS
        )
        pygame.draw.circle(
            surface=SCREEN, color=BLACK,
            center=(piece_center_x, piece_center_y), radius=PIECE_RADIUS,
            width=GRID_LINES_WIDTH
        )


def draw_pieces():
    for i in range(GRID.shape[0]):
        for j in range(GRID.shape[1]):
            if GRID[i][j] == 0: continue
            draw_piece(i, j)


def show_piece_place():
    x, y = pygame.mouse.get_pos()
    j = (x - SCREEN_PADDING) // CELL_LENGTH
    i = (y - SCREEN_PADDING) // CELL_LENGTH
    # if index out of the grid index boundary, return (None, None)
    if is_in_grid(i, j) and GRID[i][j] == 0:
        piece_center_x = SCREEN_PADDING + j*CELL_LENGTH + CELL_LENGTH//2
        piece_center_y = SCREEN_PADDING + i*CELL_LENGTH + CELL_LENGTH//2
        pygame.draw.circle(
            surface=SCREEN, color=GREY,
            center=(piece_center_x, piece_center_y), radius=PIECE_RADIUS,
            width=GRID_LINES_WIDTH
        )


def draw_go_board():
    for i in range(GRID.shape[0]):
        for j in range(GRID.shape[1]):
            top_pt = (SCREEN_PADDING + j*CELL_LENGTH + CELL_LENGTH//2, SCREEN_PADDING + i*CELL_LENGTH)
            bottom_pt = (SCREEN_PADDING + j*CELL_LENGTH + CELL_LENGTH//2, SCREEN_PADDING + (i+1)*CELL_LENGTH)
            left_pt = (SCREEN_PADDING + j*CELL_LENGTH, SCREEN_PADDING + i*CELL_LENGTH + CELL_LENGTH//2)
            right_pt = (SCREEN_PADDING + (j+1)*CELL_LENGTH, SCREEN_PADDING + i*CELL_LENGTH + CELL_LENGTH//2)
            center_pt = (SCREEN_PADDING + j*CELL_LENGTH + CELL_LENGTH//2, SCREEN_PADDING + i*CELL_LENGTH + CELL_LENGTH//2)
            if i == 0: top_pt = center_pt
            elif i == GRID.shape[0]-1: bottom_pt = center_pt
            if j == 0: left_pt = center_pt
            elif j == GRID.shape[1]-1: right_pt = center_pt
            pygame.draw.line(
                surface=SCREEN, color=BLACK,
                start_pos=top_pt, end_pos=bottom_pt,
                width=GRID_LINES_WIDTH
            )
            pygame.draw.line(
                surface=SCREEN, color=BLACK,
                start_pos=left_pt, end_pos=right_pt,
                width=GRID_LINES_WIDTH
            )


def draw_attax_board():
    pygame.draw.rect(
        surface=SCREEN, color=BLACK,
        rect=(SCREEN_PADDING, SCREEN_PADDING, SCREEN_WIDTH-2*SCREEN_PADDING, SCREEN_HEIGHT-2*SCREEN_PADDING),
        width=GRID_LINES_WIDTH
    )
    for i in range(1, GRID.shape[0]):
        start_pt = (SCREEN_PADDING, SCREEN_PADDING + i*CELL_LENGTH)
        end_pt = (SCREEN_WIDTH-SCREEN_PADDING, SCREEN_PADDING + i*CELL_LENGTH)
        pygame.draw.line(
            surface=SCREEN, color=BLACK,
            start_pos=start_pt, end_pos=end_pt,
            width=GRID_LINES_WIDTH
        )
    for j in range(1, GRID.shape[1]):
        start_pt = (SCREEN_PADDING + j*CELL_LENGTH, SCREEN_PADDING)
        end_pt = (SCREEN_PADDING + j*CELL_LENGTH, SCREEN_HEIGHT-SCREEN_PADDING)
        pygame.draw.line(
            surface=SCREEN, color=BLACK,
            start_pos=start_pt, end_pos=end_pt,
            width=GRID_LINES_WIDTH
        )


def draw_board():
    SCREEN.fill(BG_COLOR)
    if GAME_TITLE == "a" or GAME_TITLE == "attax": draw_attax_board()
    elif GAME_TITLE == "g" or GAME_TITLE == "go": draw_go_board()
    if GAME_TITLE in ['attax', 'a', 'go', 'g']: draw_pieces()


def piece_index_click():
    x, y = pygame.mouse.get_pos()
    j = (x - SCREEN_PADDING) // CELL_LENGTH
    i = (y - SCREEN_PADDING) // CELL_LENGTH
    # if index out of the grid index boundary, return (None, None)
    if not all(map(lambda index: 0 <= index and index < GRID.shape[0], [i, j])):
        return None, None
    return i, j

'''
def show_selected_piece(i: int, j: int):
    piece_center_x = SCREEN_PADDING + j*cell_length + cell_length//2
    piece_center_y = SCREEN_PADDING + i*cell_length + cell_length//2
    piece_color = (RED, BLUE) if GAME_TITLE == "attax" or GAME_TITLE == "a" else (WHITE, BLACK)
    pygame.draw.circle(
        surface=screen, color=piece_color[0] if grid[i][j] == 1 else piece_color[1],
        center=(piece_center_x, piece_center_y),
        radius=piece_radius,
    )
    pygame.draw.circle(
        surface=screen, color=BLACK,
        center=(piece_center_x, piece_center_y),
        radius=piece_radius, width=GRID_LINES_WIDTH
    )
'''

def show_graphics(game_title: Literal['attax', 'a', 'go', 'g'], grid: np.ndarray):
    pygame.init()
    SET_GLOBALS(game_title, grid)
    SET_SCREEN()
    
    # main loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                i, j = piece_index_click()
                print(f"i={i} j={j}")

        draw_board()
        show_piece_place()

        pygame.display.flip()
        

if __name__ == "__main__":
    grid = np.array([
        [0, 1, 0, 1, 0, 1, 0, 0, 2],
        [0, 0, 1, 2, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 2, 1, 1],
        [0, 2, 2, 0, 0, 0, 1, 1, 1],
        [1, 0, 2, 1, 0, 0, 0, 1, 1],
        [0, 1, 0, 1, 2, 1, 0, 0, 1],
        [0, 1, 0, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 2, 1, 1, 0, 1],
        [0, 0, 0, 2, 0, 0, 0, 0, 2]
    ])
    show_graphics("g", grid)


