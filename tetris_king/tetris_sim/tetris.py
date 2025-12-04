## okay so let's code some TETRIS FROM SCRATCH WOOOO

import numpy as np
import sys
import random
import threading 
import pygame
import pygame.freetype

# let's start with a board. 


# so we specifiy position like board[y][x], where (0,0) is top left

# and also let's just do some pieces 

I = (np.array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               [1, 1, 1, 1],
               [0, 0, 0, 0]]))      # starts flat??, spawns at (4,0)

T = (np.array([[0, 1, 0],
               [1, 1, 1],
               [0, 0, 0]]))     # starts nub top middle, spawns at (4,-1)

O = (np.array([[1, 1],
               [1, 1]]))        # doesn't matter, spawns at (5,-1)

S = (np.array([[0, 1, 1],
               [1, 1, 0],
               [0, 0, 0]]))     # starts nub top right, spawns at (4,-1)       

Z = (np.array([[1, 1, 0],
               [0, 1, 1],
               [0, 0, 0]]))     # starts nub top left, spawns at 

J = (np.array([[1, 0, 0],
               [1, 1, 1],
               [0, 0, 0]]))     # starts nub top left    

L = (np.array([[0, 0, 1],
               [1, 1, 1],
               [0, 0, 0]]))     # starts nub top right, spwans at (4,-1)

PIECE_LIST = ['I', 'T', 'O', 'S', 'Z', 'L', 'J']

PIECE_DICT = {
    'I': I, 
    'T': T,
    'O': O,
    'S': S,
    'Z': Z,
    'L': L,
    'J': J
}

LOCATIONS = {
    'I': (4, 0),
    'T': (4, 1),
    'O': (5, 1),
    'S': (4, 1),
    'Z': (4, 1),
    'L': (4, 1),
    'J': (4, 1)
}


# I saw this in chess once. Maybe this is what they mean when they
# saw learn by doing haha

class Piece:
    def __init__(self, type):
        self.type = type
        self.blocks = PIECE_DICT[type]
        self.height = self.blocks.shape[0]
        self.width = self.blocks.shape[1]

    def rotate(self, n_rotations=1):
        for _ in range(n_rotations):
            self.blocks = np.rot90(self.blocks, k=-1)


class Tetris:
    def __init__(self):
        pygame.init()

        self.WIDTH = 10
        self.HEIGHT = 20 
        self.CELL_SIZE = 30
        self.FPS = 20
        self.DROPTIME = 500  # milliseconds

        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
        self.WHITE = (255,255,255)


        self.screen = pygame.display.set_mode((self.WIDTH*self.CELL_SIZE, (self.HEIGHT+2)*self.CELL_SIZE))
        # create board with walls
        self.board = np.zeros((self.HEIGHT+3, self.WIDTH+2))
        self.board[-1, :] = 2
        self.board[:, 0] = 5
        self.board[:, -1] = 5

        # generate current piece
        self.current_piece = None
        self.piece_x = 0
        self.piece_y = 0

        self.next_pieces = [random.choice(PIECE_LIST), 
                            random.choice(PIECE_LIST), 
                            random.choice(PIECE_LIST)]

        self.key_inputs = set()
        self.clock = pygame.time.Clock()
        self.time = 0
        self.last_time = 0

        self.font = pygame.freetype.Font(None, 30)

        # run our threads (MAY NOT BE NEEDED WITH PYGAME)
        # loop_thread = threading.Thread(target=self.loop)
        # key_thread = threading.Thread(target=self.key_listener)
        # loop_thread.start()
        # key_thread.start()



    def spawn_piece(self):
        # create a new piece 
        next_piece = self.next_pieces[0]
        self.current_piece = Piece(next_piece)
        spawn_location = LOCATIONS[next_piece]
        self.piece_x = spawn_location[0]
        self.piece_y = spawn_location[1]

        # update the list 
        self.next_pieces[0] =  self.next_pieces[1] 
        self.next_pieces[1] =  self.next_pieces[2] 
        self.next_pieces[2] =  random.choice(PIECE_LIST)


    def detect_collision(self, offset_x=0, offset_y=0):
        for i in range(self.current_piece.height):
            for j in range(self.current_piece.width):
                if self.current_piece.blocks[i][j] == 1:
                    if self.board[self.piece_y + i + offset_y][self.piece_x + j + offset_x] in [2, 5]:
                        return True
        return False
    
    def check_and_clear_tetris(self):
        # let's find full rows
        tetris_rows = []
        for row in range(2, self.HEIGHT+2):
            if all(self.board[row][1:-1] == 2):
                tetris_rows.append(row)

        # then, let's clear them
        for row in tetris_rows:
            self.board[3:row+1, 1:-1] = self.board[2:row, 1:-1]
            self.board[2, 1:-1] = 0
    

    def handle_inputs(self):
        if 'a' in self.key_inputs:
            if not self.detect_collision(offset_x=-1):
                self.piece_x -= 1
            self.key_inputs.remove('a')

        if 'd' in self.key_inputs:
            if not self.detect_collision(offset_x=1):
                self.piece_x += 1
            self.key_inputs.remove('d')

        if 'r' in self.key_inputs:
            # rotate piece
            original_blocks = self.current_piece.blocks.copy()
            self.current_piece.rotate()

            # check for collision after rotation 
            # there is a WHOLE damn system for kicking a block but
            # let's ignore that for now because it seems too complicated
            if self.detect_collision():
                self.current_piece.blocks = original_blocks
            self.key_inputs.remove('r')


    def clear_piece(self):
        # clear current piece from board
        for row in range(self.HEIGHT+3):
            for cell in range(self.WIDTH+2):
                if self.board[row][cell] == 1:
                    self.board[row][cell] = 0

    def place_piece(self):
        # place current piece on board
        for i in range(self.current_piece.height):
            for j in range(self.current_piece.width):
                if self.current_piece.blocks[i][j] == 1:
                    self.board[self.piece_y + i][self.piece_x + j] = 1

    def update_board(self):
        # spawn piece if needed
        if self.current_piece is None:
            self.spawn_piece()
        
        self.clear_piece()

        self.handle_inputs()

        self.place_piece()

        # update clock and move piece down
        self.time += self.clock.get_time()
        if self.time - self.last_time > self.DROPTIME:
            # before moving down, first check for collisions 
            if self.detect_collision(offset_y=1):
                for i in range(self.current_piece.height):
                    for j in range(self.current_piece.width):
                        if self.current_piece.blocks[i][j] == 1:
                            self.board[self.piece_y + i][self.piece_x + j] = 2
                
                self.check_and_clear_tetris()
                self.current_piece = None

            else:
                self.piece_y += 1
            self.last_time = self.time

        self.clock.tick(self.FPS)


    def update_screen(self):
        for row in range(self.HEIGHT+3):
            for cell in range(self.WIDTH+2):
                if self.board[row][cell] == 0:
                    pygame.draw.rect(self.screen, self.BLACK, pygame.Rect((cell-1)*self.CELL_SIZE, (row-2)*self.CELL_SIZE, 30, 30))
                elif self.board[row][cell] == 1:
                    pygame.draw.rect(self.screen, self.RED, pygame.Rect((cell-1)*self.CELL_SIZE, (row-2)*self.CELL_SIZE, 30, 30))
                elif self.board[row][cell] == 2:
                    pygame.draw.rect(self.screen, self.BLUE, pygame.Rect((cell-1)*self.CELL_SIZE, (row-2)*self.CELL_SIZE, 30, 30))

        pygame.draw.rect(self.screen, self.WHITE, pygame.Rect(0, (self.HEIGHT)*self.CELL_SIZE, self.WIDTH*self.CELL_SIZE, 2*self.CELL_SIZE))
        self.font.render_to(self.screen, (self.CELL_SIZE*1.15, (self.HEIGHT+.5)*self.CELL_SIZE), f"Next: {self.next_pieces}", (0, 0, 0))

        pygame.display.flip()
        

    def key_listener(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Key pressed
            if event.type == pygame.KEYDOWN:
                self.key_inputs.add(pygame.key.name(event.key))
                # print(f"Key pressed: {event.key}")

    def check_loss(self):
        for cell in self.board[1]:
            if cell == 2:
                print("Game Over!")
                pygame.quit()
                sys.exit()
        
    def loop(self):
        while True:
            self.key_listener()
            self.update_board()
            self.update_screen()
            self.check_loss()


# my_tetris = Tetris()
# my_tetris.loop()
