## okay so let's code some TETRIS FROM SCRATCH WOOOO

import numpy as np
import sys
import random
import pygame
import pygame.freetype

# let's start with a board.


# so we specifiy position like board[y][x], where (0,0) is top left

# and also let's just do some pieces

I = np.array(
    [[0, 0, 0, 0], [1, 1, 1, 1],  [0, 0, 0, 0], [0, 0, 0, 0]]
)  # starts flat??, spawns at (4,0)

T = np.array(
    [[0, 1, 0], [1, 1, 1], [0, 0, 0]]
)  # starts nub top middle, spawns at (4,-1)

O = np.array([[1, 1], [1, 1]])  # doesn't matter, spawns at (5,-1)

S = np.array(
    [[0, 1, 1], [1, 1, 0], [0, 0, 0]]
)  # starts nub top right, spawns at (4,-1)

Z = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]])  # starts nub top left, spawns at

J = np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0]])  # starts nub top left

L = np.array(
    [[0, 0, 1], [1, 1, 1], [0, 0, 0]]
)  # starts nub top right, spwans at (4,-1)

PIECE_LIST = ["I", "T", "O", "S", "Z", "L", "J"]

PIECE_DICT = {"I": I, "T": T, "O": O, "S": S, "Z": Z, "L": L, "J": J}

LOCATIONS = {
    "I": (4, 0),
    "T": (4, 1),
    "O": (5, 1),
    "S": (4, 1),
    "Z": (4, 1),
    "L": (4, 1),
    "J": (4, 1),
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
        # initalize our class and all our variables
        pygame.init()

        self.WIDTH = 10
        self.HEIGHT = 20
        self.CELL_SIZE = 30
        self.FPS = 20
        self.DROPTIME = 500  # milliseconds

        # initialize colors for visualizer 
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)

        # make a screen where we can display things 
        self.screen = pygame.display.set_mode(
            (self.WIDTH * self.CELL_SIZE, (self.HEIGHT + 2) * self.CELL_SIZE)
        )
        # create board with walls
        self.board = np.zeros((self.HEIGHT + 3, self.WIDTH + 2))
        self.board[-1, :] = 2
        self.board[:, 0] = 5
        self.board[:, -1] = 5

        # generate current piece
        self.current_piece = None
        self.piece_x = 0
        self.piece_y = 0
        self.lines_cleared = 0

        # random next three pieces (for now)
        self.next_pieces = [
            random.choice(PIECE_LIST),
            random.choice(PIECE_LIST),
            random.choice(PIECE_LIST),
        ]

        self.key_inputs = set()
        self.clock = pygame.time.Clock()
        self.time = 0
        self.last_time = 0

        self.font = pygame.freetype.Font(None, 30)



    def spawn_piece(self):
        '''
        function to create a new piece and update our class variables 
        updates self.current piece based on our peice list, and then 
        updates the piece list 

        updates the x and y location of the piece based on the spawning locations
        '''
        # create a new piece
        next_piece = self.next_pieces[0]
        self.current_piece = Piece(next_piece)
        spawn_location = LOCATIONS[next_piece]
        self.piece_x = spawn_location[0]
        self.piece_y = spawn_location[1]

        # update the list
        self.next_pieces[0] = self.next_pieces[1]
        self.next_pieces[1] = self.next_pieces[2]
        self.next_pieces[2] = random.choice(PIECE_LIST)

    def detect_collision(self, offset_x=0, offset_y=0):
        '''
        function to detect a collision between the current piece and 
        the board state. if it detections a collision between a currently placed
        piece, the wall, or the floor, it returns TRUE

        it can also take in an offset (as integers), which is needed as you move the 
        peice in one direction or the other 
        '''
        for i in range(self.current_piece.height):
            for j in range(self.current_piece.width):
                if self.current_piece.blocks[i][j] == 1:
                    if self.board[self.piece_y + i + offset_y][
                        self.piece_x + j + offset_x
                    ] in [2, 5]:
                        return True
        return False

    def check_and_clear_tetris(self):
        '''
        checks the full board and removes any lines that need to be cleared, 
        then moves down the boxes above them

        this should be called after placing a peice 
        '''
        # let's find full rows
        tetris_rows = []
        for row in range(2, self.HEIGHT + 2):
            if all(self.board[row][1:-1] == 2):
                tetris_rows.append(row)

        # then, let's clear them
        for row in tetris_rows:
            self.lines_cleared += 1
            self.board[3 : row + 1, 1:-1] = self.board[2:row, 1:-1]
            self.board[2, 1:-1] = 0

    def handle_inputs(self):
        '''
        this version of tetris needs to handle user inputs. we have a keylister that checks 
        if the user presses any keys, then appends them to a list (self.key_inputs). This 
        function checks the list and performs the needed action before removing the key from the list

        possible actions: 
        a -> move left 
        d -> move right 
        r -> rotate clockwise 
        '''
        if "a" in self.key_inputs:
            if not self.detect_collision(offset_x=-1):
                self.piece_x -= 1
            self.key_inputs.remove("a")

        if "d" in self.key_inputs:
            if not self.detect_collision(offset_x=1):
                self.piece_x += 1
            self.key_inputs.remove("d")

        if "r" in self.key_inputs:
            # rotate piece
            original_blocks = self.current_piece.blocks.copy()
            self.current_piece.rotate()

            # check for collision after rotation
            # there is a WHOLE damn system for kicking a block but
            # let's ignore that for now because it seems too complicated
            if self.detect_collision():
                self.current_piece.blocks = original_blocks
            self.key_inputs.remove("r")

    def clear_piece(self):
        '''
        a simple function that removes the current piece from the board matrix 
        by turning the values of the current piece (represented by 1s) to 0s
        '''
        # clear current piece from board
        for row in range(self.HEIGHT + 3):
            for cell in range(self.WIDTH + 2):
                if self.board[row][cell] == 1:
                    self.board[row][cell] = 0

    def place_piece(self):
        '''
        this function places the piece on the board at its current location
        working in conjunction with the clear_piece and handle_inputs function to move the 
        piece on the board. 
        '''
        # place current piece on board
        for i in range(self.current_piece.height):
            for j in range(self.current_piece.width):
                if self.current_piece.blocks[i][j] == 1:
                    self.board[self.piece_y + i][self.piece_x + j] = 1

    def update_board(self):
        '''
        this is the function to update the board each timestep.
        it checks for user input and checks if the piece should be moved, 
        then updates accordingly. 

        if the piece collides with the ground or another piece when moving down, 
        it sets it in place and checks for tetrises 

        if there isn't a current piece, then it spawns a new one
        '''
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
        '''
        function to display the board 
        basically, iterates through each cell in the board and 
        colors it either black (background), blue (placed blocks),
        or red (current piece)

        it also renders the next three blocks as their respective letters 
        '''
        for row in range(self.HEIGHT + 3):
            for cell in range(self.WIDTH + 2):
                if self.board[row][cell] == 0:
                    pygame.draw.rect(
                        self.screen,
                        self.BLACK,
                        pygame.Rect(
                            (cell - 1) * self.CELL_SIZE,
                            (row - 2) * self.CELL_SIZE,
                            30,
                            30,
                        ),
                    )
                elif self.board[row][cell] == 1:
                    pygame.draw.rect(
                        self.screen,
                        self.RED,
                        pygame.Rect(
                            (cell - 1) * self.CELL_SIZE,
                            (row - 2) * self.CELL_SIZE,
                            30,
                            30,
                        ),
                    )
                elif self.board[row][cell] == 2:
                    pygame.draw.rect(
                        self.screen,
                        self.BLUE,
                        pygame.Rect(
                            (cell - 1) * self.CELL_SIZE,
                            (row - 2) * self.CELL_SIZE,
                            30,
                            30,
                        ),
                    )

        pygame.draw.rect(
            self.screen,
            self.WHITE,
            pygame.Rect(
                0,
                (self.HEIGHT) * self.CELL_SIZE,
                self.WIDTH * self.CELL_SIZE,
                2 * self.CELL_SIZE,
            ),
        )
        self.font.render_to(
            self.screen,
            (self.CELL_SIZE * 1.15, (self.HEIGHT + 0.5) * self.CELL_SIZE),
            f"Next: {self.next_pieces}",
            (0, 0, 0),
        )

        pygame.display.flip()

    def key_listener(self):
        '''
        pygame keylister that checks for user input and updates the 
        self.key_inputs list 

        also checks if the user quits 
        '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Key pressed
            if event.type == pygame.KEYDOWN:
                self.key_inputs.add(pygame.key.name(event.key))
                # print(f"Key pressed: {event.key}")

    def check_loss(self):
        '''
        checks if a peice has been placed above the board (in row 1)
        if a piece has been places that high, then the game is over 
        and this function returns True
        '''
        for cell in self.board[1]:
            if cell == 2:
                # print("Game Over!")
                # pygame.quit()
                return True
                # sys.exit()
        return False

    def loop(self):
        '''
        a loop which iterates through the different steps 
        first, we check for input with self.key_listener 
        next, we update the board based on the input    
            updating the board also checks for tetrises and iterates to 
            the next piece 
        next, we update the screen
        lastly, we check if we've lost 
        '''
        while True:
            self.key_listener()
            self.update_board()
            self.update_screen()
            self.check_loss()


def main(args=None):
    '''
    main functions which initialized an instance of the tetris class 
    and loops it 
    '''
    my_tetris = Tetris()
    my_tetris.loop()

if __name__ == "__main__":
    main()




