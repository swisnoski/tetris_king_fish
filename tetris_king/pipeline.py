from tetris_sim.tetris_pipeline import Tetris_PIPE
from tetris_sim.tetris_max import find_best_move
from cv_tetris.cv_pipeline import initialize_video_capture, get_cv_info
import numpy as np



if __name__ == "__main__":
    cap = initialize_video_capture()
    last_piece = "I_PIECE"
    game_state, current_piece = get_cv_info(cap, last_piece)


def main():
    my_tetris = Tetris_PIPE()
    cap = initialize_video_capture()

    while True:
            current_piece = None
            while current_piece is None:
                game_state, current_piece = get_cv_info(cap, my_tetris.current_piece.type)

            try:
                np.testing.assert_array_equal(my_tetris.board[2:-1,1:-1], game_state)
            except: 
                print("Discrepancy between CV detected board and internal board!")
                print(f"CV detected board:\n{game_state}")
                print(f"Internal board:\n{my_tetris.board[2:-1,1:-1]}")
                my_tetris.board[2:-1,1:-1] = game_state

            my_tetris.update_piece(current_piece)
            r, t = find_best_move(my_tetris.board, my_tetris.current_piece.type)
            my_tetris.execute_moves(r, t) #update the board, no need to display

            # somehow communicate the move to the robot here






if __name__ == "__main__":
    main()
