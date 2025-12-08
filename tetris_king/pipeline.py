import numpy as np
import rclpy
from rclpy.node import Node
from threading import Thread

from my_interfaces.msg import MyTuple  

from tetris_sim.tetris_pipeline import Tetris_PIPE
from tetris_sim.tetris_max import find_best_move
from cv_tetris.cv_pipeline import initialize_video_capture, get_cv_info





class TetrisBot(Node):
    """
    Class to initialize and establish subscriber/publisher interaction
    """
    def __init__(self):
        super().__init__('tetris_bot')
        self.action_publisher = self.create_publisher(MyTuple, '/action', 10)
        self.my_tetris = Tetris_PIPE()
        self.cap = initialize_video_capture()

        thread = Thread(target=self.loop, daemon=True)
        thread.start()
         
    def loop(self):
        while True:
                current_piece = None
                while current_piece is None:
                    game_state, current_piece = get_cv_info(self.cap)

                try:
                    np.testing.assert_array_equal(self.my_tetris.board[2:-1,1:-1], game_state)
                except: 
                    print("Discrepancy between CV detected board and internal board!")
                    print(f"CV detected board:\n{game_state}")
                    print(f"Internal board:\n{self.my_tetris.board[2:-1,1:-1]}")
                    self.my_tetris.board[2:-1,1:-1] = game_state

                self.my_tetris.update_piece(current_piece)
                r, t = find_best_move(self.my_tetris.board, self.my_tetris.current_piece.type)
                self.my_tetris.execute_moves(r, t) #update the board, no need to display


                move = MyTuple()
                move.rotation = r
                move.translation = t
                self.action_publisher.publish(move)


                # somehow communicate the move to the robot here



def main(args=None):
    rclpy.init(args=args)
    node = TetrisBot()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()