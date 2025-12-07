from tetris import Tetris, Piece, LOCATIONS
from tetris_max import find_best_move
import numpy as np



class Tetris_PIPE(Tetris):
    def __init__(self):
        # super().__init__()
        self.lines_cleared = 0
        self.iteration = 0
        self.last_score = 0

        self.WIDTH = 10
        self.HEIGHT = 20
        self.CELL_SIZE = 30
        self.FPS = 20
        self.DROPTIME = 500  # milliseconds

        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)

        # create board with walls
        self.board = np.zeros((self.HEIGHT + 3, self.WIDTH + 2))
        self.board[-1, :] = 2
        self.board[:, 0] = 5
        self.board[:, -1] = 5

        # generate current piece
        self.current_piece = None
        self.piece_x = 0
        self.piece_y = 0

        # self.next_pieces = [None, None, None]
        self.next_pieces = ["T", "T", 'T']



    def update_piece(self, type):
        # create a new piece
        self.current_piece = Piece(type)
        spawn_location = LOCATIONS[type]
        self.piece_x = spawn_location[0]
        self.piece_y = spawn_location[1]



    def execute_moves(self, r, t):
        self.current_piece.rotate(n_rotations=r)
        self.piece_x += t

        piece_in_place = False
        while piece_in_place is False:
            if self.detect_collision(offset_y=1):
                for i in range(self.current_piece.height):
                    for j in range(self.current_piece.width):
                        if self.current_piece.blocks[i][j] == 1:
                            try:
                                self.board[self.piece_y + i][self.piece_x + j] = 2
                            except IndexError:
                                print(
                                    f"ERROR. r: {r}, t: {t}, piece: {self.current_piece.type}"
                                )
                                exit(0)
                piece_in_place = True
                self.check_and_clear_tetris()
            else:
                self.piece_y += 1
