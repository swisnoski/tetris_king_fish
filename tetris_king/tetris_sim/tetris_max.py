from .tetris import Tetris, Piece, LOCATIONS
import numpy as np
import pygame
import sys



# define possibilites for each piece 
POSSIBILITIES = {
    "I": (range(2), range(-4, 6, 1)),
    "T": (range(4), range(-4, 5, 1)),
    "O": (range(1), range(-4, 5, 1)),
    "S": (range(2), range(-4, 5, 1)),
    "Z": (range(2), range(-4, 5, 1)),
    "L": (range(4), range(-4, 5, 1)),
    "J": (range(4), range(-4, 5, 1)),
}


class Tetris_MAX(Tetris):
    '''
    tetris_max is designed to use a simple heurstic to make it's moves 
    it inherits from our base tetris class and keeps most of it's 
    original logic, but swaps out the ability to play tetris with 
    an automover 
    '''
    def __init__(self):
        '''
        initalizing all our varaibles 
        we need to add a move_time to determine 
        the length between each move our algorithm decides 
        one move = one rotation or one left/right/down movement 
        '''
        super().__init__()
        self.last_move_time = 0
        self.MOVETIME = 100
        self.iteration = 0

    def auto_loop(self):
        '''
        our "auto_loop" function
        first, we spawn a new peice 
        then, we find the best possible move given our new piece and the current board state 
        lastly, we execute that move to update our board state 
        '''
        while True:
            self.spawn_piece()
            # print(self.board)
            r, t = find_best_move(self.board, self.current_piece.type)

            # print(t, r)
            self.execute_moves(r, t)  # display_board is called within this function

    def execute_moves(self, r, t):
        '''
        given a rotation (int between 0 and 3) and a 
        translation (int between -5 and 5), we rotate and 
        move our current piece accordingly. Once above it's desired spot, we 
        then drop it down. Each of these actions is contingent on the timer 

        this function places the block, checks for tetrises, and updates the screen 
        '''
        # loop until
        piece_in_place = False
        while piece_in_place is False:
            self.time += self.clock.get_time()
            self.clear_piece()

            # check if move
            if self.time - self.last_move_time > self.MOVETIME:
                if r > 0:
                    self.current_piece.rotate()
                    r -= 1
                    self.place_piece()

                elif t > 0:
                    self.piece_x += 1
                    t -= 1
                    self.place_piece()

                elif t < 0:
                    self.piece_x -= 1
                    t += 1
                    self.place_piece()

                else:
                    if self.detect_collision(offset_y=1):
                        for i in range(self.current_piece.height):
                            for j in range(self.current_piece.width):
                                if self.current_piece.blocks[i][j] == 1:
                                    self.board[self.piece_y + i][self.piece_x + j] = 2
                        piece_in_place = True
                        self.check_and_clear_tetris()
                    else:
                        self.piece_y += 1
                        self.place_piece()

                self.last_move_time = self.time
                self.update_screen()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.clock.tick(500)





def detect_collision_normal(board, piece, x, y):
    ''' 
    function to detect a collision between the current piece and 
    the board state. if it detections a collision between a currently placed
    piece, the wall, or the floor, it returns TRUE

    it can also take in an offset (as integers), which is needed as you move the 
    peice in one direction or the other 
        
    basically a copy of the class function defined in tetris.py 

    however, it can be used outside of the class 
    '''
    for i in range(piece.height):
        for j in range(piece.width):
            if piece.blocks[i][j] == 1:
                if board[y + i][x + j] in [2, 5]:
                    return True
    return False


def score_board(board, HEIGHT=20, WIDTH=10):
    '''
    takes in a tetris board (numpy array and returns a score 
    it scores the board based on four variables: 
    1. number of holes
    2. bumpiness of the surface 
    3. how high the blocks have gotten 
    4. if placing the block creates a tetris (or tetrises) 

    each of these has a designated weight as shown below: 
    score = (
        (total_height * -0.510066)
        + (lines_cleared * 0.760666)
        + (holes * -0.35663)
        + (bumpiness * -0.184483)
    )
    '''
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


# def score_board_normal(board, HEIGHT=20, WIDTH=10):


def find_best_move(board, piece_type):
    '''
    this functions iterates through all possibilites given 
    a board (a numpy array) and a given piece type (string, single leter 
    representing a tetris piece). 

    pulling from the "possibilites" dict defined at the top of this file, 
    we go through each rotation, then move the piece left and right, drop it, 
    and calculate the score of the updated board. the highest score and the respective 
    moves to achieve it are saved, and we eventually return the idea rotation and translation
    '''
    # should need board, piece, possibilities
    top_score = -float("inf")
    run_score = 0
    top_position = (0, 0)

    pos_rotations, pos_translations = POSSIBILITIES[piece_type]
    # print("here 1")

    for r in pos_rotations:
        # print("running rot")
        # initialize the piece
        testing_piece = Piece(piece_type)
        # give it a spin
        testing_piece.rotate(n_rotations=r)

        # give it a move
        for t in pos_translations:
            # print("running t")

            # reset the locations
            testing_piece_x, testing_piece_y = LOCATIONS[piece_type]

            # reset the board
            testing_board = np.copy(board)

            # move the piece over
            testing_piece_x += t

            # for now, if it doesn't collide ABOVE the space, we are just going to assume that it can make it
            # to the location safely

            if not detect_collision_normal(
                testing_board, testing_piece, testing_piece_x, testing_piece_y
            ):
                # so if we've made it this far, we can drop the piece. notably, we don't actually want to
                # visualize this at all, we just are doing calculations here for now

                drop_value = 0
                while not detect_collision_normal(
                    testing_board,
                    testing_piece,
                    testing_piece_x,
                    testing_piece_y + drop_value,
                ):
                    drop_value += 1
                drop_value -= 1

                # place piece on board
                for i in range(testing_piece.height):
                    for j in range(testing_piece.width):
                        if testing_piece.blocks[i][j] == 1:
                            testing_board[testing_piece_y + drop_value + i][
                                testing_piece_x + j
                            ] = 2

                # lastly, calculate the score
                run_score = score_board(testing_board)
                if run_score > top_score:
                    top_position = (r, t)
                    top_score = run_score
    # print("here 2")

    return top_position


def main():
    '''
    main function, creates a tetris_max instance and then
    loops it 
    '''
    my_tetris_max = Tetris_MAX()
    my_tetris_max.auto_loop()


if __name__ == "__main__":
    main()
