from .tetris import Tetris, Piece, LOCATIONS
import numpy as np
import random

POSSIBILITIES = {
    "I": (range(2), range(-4, 6, 1)),
    "T": (range(4), range(-4, 5, 1)),
    "O": (range(1), range(-4, 5, 1)),
    "S": (range(2), range(-4, 5, 1)),
    "Z": (range(2), range(-4, 5, 1)),
    "L": (range(4), range(-4, 5, 1)),
    "J": (range(4), range(-4, 5, 1)),
}

PIECE_LIST = ["I", "T", "O", "S", "Z", "L", "J"]

PIECE_ENCODING = {
    "I": [1, 0, 0, 0, 0, 0, 0],
    "T": [0, 1, 0, 0, 0, 0, 0],
    "O": [0, 0, 1, 0, 0, 0, 0],
    "S": [0, 0, 0, 1, 0, 0, 0],
    "Z": [0, 0, 0, 0, 1, 0, 0],
    "L": [0, 0, 0, 0, 0, 1, 0],
    "J": [0, 0, 0, 0, 0, 0, 1],
}


class Tetris_RL(Tetris):
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

        self.next_pieces = [
            random.choice(PIECE_LIST),
            random.choice(PIECE_LIST),
            random.choice(PIECE_LIST),
        ]


    def initialize(self):
        self.spawn_piece()
        # print(self.current_piece.type)
        valid_moves = find_legal_moves(self.board, self.current_piece.type)
        state = self.state_returner()
        return state, valid_moves


    def reset(self):
        self.board = np.zeros((self.HEIGHT + 3, self.WIDTH + 2))
        self.board[-1, :] = 2
        self.board[:, 0] = 5
        self.board[:, -1] = 5

        self.current_piece = None
        self.lines_cleared = 0
        self.iteration = 0


    def step(self, action):

        current_score = score_board(self.board)

        r = (action - 1) // 11
        t = (action - 1) % 11 - 5

        # print(f"r: {r}, t: {t}, piece: {self.current_piece.type}")

        self.execute_moves(r, t)

        self.spawn_piece()

        next_state = self.state_returner()

        new_score = score_board(self.board)

        reward = new_score - current_score

        done = self.check_loss()
        valid_moves = find_legal_moves(self.board, self.current_piece.type)
        if all(v == 0 for v in valid_moves):
            done = True

        self.iteration += 1  # Placeholder for iteration count if needed

        # print("-----------------------------------")
        # print(f"Next State:\n{self.board}")
        # print(
        #     f"Step: {self.iteration}, Reward: {reward}, Done: {done}, Lines Cleared: {self.lines_cleared}"
        # )

        return next_state, reward, done, self.iteration, valid_moves

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

    def state_returner(self):
        column_counts = []
        for col in range(1, 11):
            count = np.sum(self.board[:, col] == 2) - 1
            column_counts.append(count)
        column_counts.extend(PIECE_ENCODING[self.current_piece.type])
        for next_piece in self.next_pieces:
            column_counts.extend(PIECE_ENCODING[next_piece])
        return column_counts


def detect_collision_normal(board, piece, x, y):
    for i in range(piece.height):
        for j in range(piece.width):
            if piece.blocks[i][j] == 1:
                if board[y + i][x + j] in [2, 5]:
                    return True
    return False


def score_board(board, HEIGHT=20, WIDTH=10):
    holes = 0
    hole_detector = False
    col_heights = []
    col_height = HEIGHT

    # loop through the columns
    for column in range(1, WIDTH + 1):
        # define column and reset variables
        col = board[2:-1, column]
        hole_detector = False
        col_height = 20

        # loop through cells
        for cell in col:
            if hole_detector is False:
                if cell == 0:
                    col_height -= 1
                else:
                    hole_detector = True
            else:
                if cell == 0:
                    holes += 1

        col_heights.append(col_height)

    total_height = sum(col_heights)
    bumpiness = 0
    for i in range(len(col_heights) - 1):
        bumpiness += abs(col_heights[i] - col_heights[i + 1])

    lines_cleared = 0
    for row in range(2, HEIGHT + 2):
        if all(board[row][1:-1] != 0):
            lines_cleared += 1

    # score = [Sum Height, Lines Cleared, Holes, Bumpiness] x [-0.510066, 0.760666, -0.35663, -0.184483]
    # print(f"Total Height: {total_height}, Lines_Cleared: {lines_cleared}, Holes: {holes}, Bumps: {bumpiness}")
    score = (
        (total_height * -0.510066)
        + (lines_cleared * 0.760666)
        + (holes * -0.35663)
        + (bumpiness * -0.184483)
    )
    return score


def find_legal_moves(board, piece_type):
    # should need board, piece, possibilities

    pos_rotations, _ = POSSIBILITIES[piece_type]
    pos_translations = range(-5, 6)

    legal_moves = []

    for r in pos_rotations:
        testing_piece = Piece(piece_type)
        testing_piece.rotate(n_rotations=r)

        for t in pos_translations:
            testing_piece_x, testing_piece_y = LOCATIONS[piece_type]
            testing_board = np.copy(board)
            testing_piece_x += t

            if detect_collision_normal(
                testing_board, testing_piece, testing_piece_x, testing_piece_y
            ):
                legal_moves.append(0)
            else:
                legal_moves.append(1)

    num_rot = len(pos_rotations)
    extra_zeros = 11 * (4 - num_rot)

    for _ in range(extra_zeros):
        legal_moves.append(0)

    if len(legal_moves) != 44:
        print("WHAT THE FUCK")
        raise InterruptedError

    return legal_moves
