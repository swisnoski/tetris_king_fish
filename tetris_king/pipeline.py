import numpy as np
import cv2 as cv


from threading import Thread
from tetris_sim.tetris_pipeline import Tetris_PIPE
from tetris_sim.tetris_max import find_best_move
from cv_tetris.cv_pipeline import initialize_video_capture, initialize_grid, get_cv_info


def loop():
    my_tetris = Tetris_PIPE()
    cap = initialize_video_capture()
    grid_pts = initialize_grid(cap)
    while True:
            current_piece = None
            while current_piece is None:
                game_state, current_piece = get_cv_info(cap, grid_pts)
                if cv.waitKey(5) == ord('q'):
                    break
            print(f"{current_piece}")
            try:
                np.testing.assert_array_equal(my_tetris.board[2:-1,1:-1], game_state)
            except: 
                # print("Discrepancy between CV detected board and internal board!")
                # print(f"CV detected board:\n{game_state}")
                # print(f"Internal board:\n{my_tetris.board[2:-1,1:-1]}")
                my_tetris.board[2:-1,1:-1] = game_state

            my_tetris.update_piece(current_piece)
            r, t = find_best_move(my_tetris.board, my_tetris.current_piece.type)
            my_tetris.execute_moves(r, t) #update the board, no need to display

            # input('yes')
            # if cv.waitKey(0) == ord('y'):
            #     continue
            

def main(args=None):
    loop()

if __name__ == "__main__":
    main()
