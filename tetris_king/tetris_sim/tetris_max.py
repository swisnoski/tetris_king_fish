from tetris import Tetris, Piece, LOCATIONS
import numpy as np
import pygame
import sys


POSSIBILITIES = {
    'I': (range(2), range(-4,6,1)),
    'T': (range(4), range(-4,5,1)),
    'O': (range(1), range(-4,5,1)),
    'S': (range(2), range(-4,5,1)),
    'Z': (range(2), range(-4,5,1)),
    'L': (range(4), range(-4,5,1)),
    'J': (range(4), range(-4,5,1)),
}


class Tetris_MAX(Tetris):
    def __init__(self):
        super().__init__()
        self.last_move_time = 0 
        self.MOVETIME = 50
        self.iteration = 0


    def auto_loop(self):
        while True:
            self.spawn_piece()

            r, t = find_best_move(self.board, self.current_piece.type)

            # print(t, r)
            self.execute_moves(r, t) # display_board is called within this function 

    def auto_step(self):
        while True:
            self.step()


    def step(self, action=None):

        current_score = score_board(self.board)

        self.spawn_piece()

        # r, t = action
        r, t = find_best_move(self.board, self.current_piece.type)

        self.execute_moves(r, t)

        next_state = np.copy(self.board)

        new_score = score_board(self.board)
        reward = new_score - current_score

        done = self.check_loss()
        self.iteration += 1  # Placeholder for iteration count if needed

        print("-----------------------------------")
        print(f"Next State:\n{self.board}")
        print(f"Step: {self.iteration}, Reward: {reward}, Done: {done}")

        valid_moves = find_legal_moves(self.board, self.next_pieces[0])



        return next_state, reward, done, self.iteration, valid_moves


    def execute_moves(self, r, t):
        # loop until
        piece_in_place = False
        while piece_in_place is False: 
            self.time += self.clock.get_time()
            self.clear_piece()

            # check if move 
            if self.time - self.last_move_time > self.MOVETIME:
                if r > 0:
                    self.current_piece.rotate()
                    r-=1
                    self.place_piece()

                elif t > 0: 
                    self.piece_x += 1
                    t-=1
                    self.place_piece()

                elif t < 0:
                    self.piece_x -= 1
                    t+=1
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
            self.clock.tick(self.FPS)

    
            

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
        for column in range(1, WIDTH+1):
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
            bumpiness += abs(col_heights[i] - col_heights[i+1])


        lines_cleared = 0
        for row in range(2, HEIGHT+2):
            if all(board[row][1:-1] != 0):
                lines_cleared += 1

        # score = [Sum Height, Lines Cleared, Holes, Bumpiness] x [-0.510066, 0.760666, -0.35663, -0.184483]
        # print(f"Total Height: {total_height}, Lines_Cleared: {lines_cleared}, Holes: {holes}, Bumps: {bumpiness}")
        score = (total_height * -0.510066) + (lines_cleared * 0.760666) + (holes * -0.35663) + (bumpiness * -0.184483)
        return score


# def score_board_normal(board, HEIGHT=20, WIDTH=10):

    

def find_best_move(board, piece_type):
    # should need board, piece, possibilities
    top_score = -float("inf")
    run_score = 0
    top_position = (0,0)

    pos_rotations, pos_translations = POSSIBILITIES[piece_type]

    for r in pos_rotations:
        # initialize the piece 
        testing_piece = Piece(piece_type)
        # give it a spin
        testing_piece.rotate(n_rotations=r)

        # give it a move
        for t in pos_translations:

            # reset the locations
            testing_piece_x, testing_piece_y = LOCATIONS[piece_type]
            
            # reset the board 
            testing_board = np.copy(board)

            # move the piece over 
            testing_piece_x += t

            # for now, if it doesn't collide ABOVE the space, we are just going to assume that it can make it 
            # to the location safely

            if not detect_collision_normal(testing_board, testing_piece, testing_piece_x, testing_piece_y):
                # so if we've made it this far, we can drop the piece. notably, we don't actually want to 
                # visualize this at all, we just are doing calculations here for now 

                drop_value = 0
                while not detect_collision_normal(testing_board, testing_piece, testing_piece_x, testing_piece_y+drop_value): 
                    drop_value += 1
                drop_value -= 1

                # place piece on board
                for i in range(testing_piece.height):
                    for j in range(testing_piece.width):
                        if testing_piece.blocks[i][j] == 1:
                            testing_board[testing_piece_y + drop_value + i][testing_piece_x + j] = 2

                # lastly, calculate the score
                run_score = score_board(testing_board)
                if run_score > top_score:
                    top_position = (r, t)
                    top_score = run_score
    
    return top_position




def find_legal_moves(board, piece_type):
    # should need board, piece, possibilities

    pos_rotations, _ = (POSSIBILITIES[piece_type])
    pos_translations = range(-5,6)

    legal_moves = []

    for r in pos_rotations:
        testing_piece = Piece(piece_type)
        testing_piece.rotate(n_rotations=r)

        for t in pos_translations:
            testing_piece_x, testing_piece_y = LOCATIONS[piece_type]
            testing_board = np.copy(board)
            testing_piece_x += t

            if not detect_collision_normal(testing_board, testing_piece, testing_piece_x, testing_piece_y):
                legal_moves.append(1)
            else:
                legal_moves.append(0)


    num_rot = len(pos_rotations)
    extra_zeros = 11 * (4 - num_rot)

    for _ in range(extra_zeros):
        legal_moves.append(0)

    return legal_moves




def find_best_move_MAX(board, piece_list):
    best_move = None
    best_score = -float("inf")

    for piece_type in piece_list:
        r, t = find_best_move(board, piece_type)
        # simulate the move
        testing_piece = Piece(piece_type)
        testing_piece.rotate(n_rotations=r)
        testing_piece_x, testing_piece_y = LOCATIONS[piece_type]
        testing_piece_x += t

        testing_board = np.copy(board)

        drop_value = 0
        while not detect_collision_normal(testing_board, testing_piece, testing_piece_x, testing_piece_y+drop_value): 
            drop_value += 1
        drop_value -= 1

        for i in range(testing_piece.height):
            for j in range(testing_piece.width):
                if testing_piece.blocks[i][j] == 1:
                    testing_board[testing_piece_y + drop_value + i][testing_piece_x + j] = 2

        run_score = score_board(testing_board)
        if run_score > best_score:
            best_score = run_score
            best_move = (piece_type, r, t)

    return best_move




my_tetris_rl = Tetris_MAX()
my_tetris_rl.auto_step()