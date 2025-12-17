from time import sleep
import numpy as np
import cv2 as cv
import socket

from threading import Thread
from tetris_sim.tetris_pipeline import Tetris_PIPE
from tetris_sim.tetris_max import find_best_move
from cv_tetris.cv_pipeline import initialize_video_capture, initialize_grid, get_cv_info
import random


def loop():
    """
    Overall loop to run program pipeline.
    """
    # Connect to server
    SERVER_IP = "192.168.10.2"  # Change to server's IP
    SERVER_PORT = 5000
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_IP, SERVER_PORT))
    print("Connected to server.")

    # Tetris pipeline
    my_tetris = Tetris_PIPE()
    cap = initialize_video_capture()
    grid_pts = initialize_grid(cap)
    last_list = []
    while True:
        current_piece_list = []
        select_piece_and_game = False
        current_piece = None
        game_state_current = np.zeros_like(my_tetris.board[2:-1, 1:-1])
        while not select_piece_and_game:
            game_state, current_piece = get_cv_info(cap, grid_pts)
            if current_piece is not None:
                current_piece_list.append(current_piece)
                game_state_current += game_state
            # buffer to get last detected piece due to threading bug
            if len(current_piece_list) == 3:
                current_piece = current_piece_list[-1]
                game_state_current[game_state_current < 2] = 0
                game_state_current[game_state_current > 2] = 2
                game_state = game_state_current
                select_piece_and_game = True
            # print(current_piece_list)

            if cv.waitKey(5) == ord("q"):
                break

        print(f"{current_piece}")
        if last_list == current_piece_list:
            continue
        else:
            last_list = current_piece_list
        try:
            np.testing.assert_array_equal(my_tetris.board[2:-1, 1:-1], game_state)
        except:
            print("Discrepancy between CV detected board and internal board!")
            # print(f"CV detected board:\n{game_state}")
            # print(f"Internal board:\n{my_tetris.board[2:-1,1:-1]}")
            my_tetris.board[2:-1, 1:-1] = game_state

        my_tetris.update_piece(current_piece)
        print(my_tetris.current_piece.type)
        print(my_tetris.board[2:-1, 1:-1])
        r, t = find_best_move(my_tetris.board, my_tetris.current_piece.type)
        my_tetris.execute_moves(r, t)  # update the board, no need to display
        print(f"Rotation: {r}, Translations: {t}")

        message = str([r, t])
        client_socket.sendall(message.encode())

        # data = client_socket.recv(1024)
        # print(f"Server replies: {data.decode()}")


def main(args=None):
    loop()


if __name__ == "__main__":
    main()
