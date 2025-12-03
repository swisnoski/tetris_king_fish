from tetris import Tetris



class Tetris_RL(Tetris):
    def __init__(self):
        super().__init__()
        self.child_attribute = 'I am a child'

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




my_tetris_rl = Tetris_RL()
my_tetris_rl.loop()