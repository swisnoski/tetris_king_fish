# 1) get initialized matrix from grid_detection
# 2) matrix per each grid square, check the color 
    # will have to account for outline of current block
# 3) detect current piece and hold pieces

import cv2 as cv 
import numpy as np
from typing import Optional, List, Dict, Tuple

# --- Grid constants ---
GRID_WIDTH = 10
GRID_HEIGHT = 20
# Output size for warped grid (50px/block)
WARPED_WIDTH = 50 * GRID_WIDTH   # 500 pixels wide
WARPED_HEIGHT = 50 * GRID_HEIGHT # 1000 pixels tall
# Normalize grid view

# Block colors (BGR)
# TUNE these colors!
# OpenCV uses BGR
COLOR_MAP = {
    "EMPTY": [0, 0, 0],         # Background
    "I_PIECE": [255, 0, 0],     # Blue
    "J_PIECE": [255, 255, 0],   # Cyan/Yellow
    "L_PIECE": [0, 165, 255],   # Orange
    "O_PIECE": [0, 255, 255],   # Yellow
    "S_PIECE": [0, 255, 0],     # Green
    "T_PIECE": [128, 0, 128],   # Purple
    "Z_PIECE": [0, 0, 255],     # Red
}

IMG = "./assets/start_tetris_cleaned.jpg" # dummy image path for now

def get_matrix_fill(grid_pts):
    """
    Returns a matrix (2D array) of 0 and 1s for filled game spaces.

    Args: 
        - a list of four coordinates points ex. [(y1, x1), (y2, x2), ...],
        in (y,x) coordinates to reflect opencv

    Returns:
        - a 2D array of 0s and 1s (0 empty, 1 filled) ex. [[0, 1,]]
    """
    # ----------
     # TO CONSIDER --> THIS LOWK JUST NEEDS TO RUN ONCE (JUST GRID?)
    # make grid of pixel coord pts to check screen image [[(x, y)]]
    check_grid = None
    # check_grid = [[0 for _ in GRID_WIDTH] for _ in GRID_HEIGHT]

    # use dummy data first -- assume start left pt, clockwise order
    left_x = grid_pts[0][0]
    right_x = grid_pts[1][0] 
    up_y = grid_pts[0][1]
    bottom_y = grid_pts[2][1]
    # calculate the pixel center points to fill grid 
    width = right_x - left_x # upper right x - upper left x
    height = up_y - bottom_y  # upper left y - bottom left y 
    cell_length = width / GRID_WIDTH # width / grid_num (10)
    offset_cell = cell_length / 2 # half of cell to offset to center

    # fill matrix with coords of each cell's center point 
    img = cv.imread(IMG)
    for i in range(GRID_WIDTH):
        for j in range(GRID_HEIGHT):
            # set coordinate point
            y, x = up_y - (j * cell_length + offset_cell), left_x + (i * cell_length - offset_cell) 
            check_grid[i][j] = y, x
            # verify for now with drawing point on image
            cv.circle(img, (x, y), 5, (255, 255, 255), -1) 
    show_close()

    # ----------
    # output_grid = check_fill(check_grid)

    # return output_grid

def check_fill(check_grid):
    """
    Checks the color of each grid cell by pixel coordinate to see if filled

    Returns output_grid of 0's and 1's
    """
    # populate this grid to hold 0 and 1's, should have 200 (grid cell count)
    output_grid = [[0 for _ in GRID_WIDTH] for _ in GRID_HEIGHT]

    # iterate through the cetto check the game piece
    for i, row in enumerate(check_grid):
        for i, col in enumerate(row):
            # check color 
            # pixel = img[row, col]
            y, x = col
            pixel = img[y, x]
            # check that any BGR over threhold --> colored
            if any(channel > 50 for channel in pixel):
             # access image at row pixel, col pixel
                output_grid[i][i] = 1
    
    return output_grid
    
    # if piece_type != "EMPTY":
    #               # White dots on pieces.
    #                 cv.circle(warped_grid, (center_x, center_y), 5, (255, 255, 255), -1) 

def get_player_pieces():
    """
    Returns a list containing current and next pieces
    """

# --- Grid detect thresholds ---
# TUNE BGR to find grid lines.
# Used in cv.inRange.
LOWER_THRESHOLD = np.array([200, 200, 200]) # BGR light colors (placeholder)
UPPER_THRESHOLD = np.array([255, 255, 255]) # Max BGR (placeholder)

# Color tolerance (higher=easier match)
COLOR_TOLERANCE = 30 
# Squared for distance check.

def classify_cell_color(bgr_color: np.ndarray) -> str:
    """Finds closest color in COLOR_MAP."""
    min_distance = float('inf')
    closest_color_name = "EMPTY"
    
    for name, color in COLOR_MAP.items():
        color_array = np.array(color)
        
        # Squared distance (B,G,R space).
        distance = np.sum((bgr_color.astype(int) - color_array.astype(int)) ** 2)
        
        if distance < min_distance:
            min_distance = distance
            closest_color_name = name
    
    # Check if distance OK.
    if closest_color_name != "EMPTY" and min_distance < COLOR_TOLERANCE ** 2:
        return closest_color_name
    else:
        return "EMPTY"
        
def show_close(caption, img):
    """
    Helper function to simplify repeated showing and closing of cv windows

    caption: a string for the img caption
    img: CV image to show
    """
    cv.imshow(caption, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def main(args=None):
    
    # --- TESTING LOGIC ---
    # CHANGE THIS PATH to the location of your Tetris screenshot/photo!
    sample_image_path = 'Tetris_game_ss.png'

    # Load, run detect once.
    frame_to_process = cv.imread(sample_image_path)
    # if frame_to_process is  not None:
    #     print(f"Successfully loaded image: {sample_image_path}. Running detection...")
    #     process_image(frame_to_process)
    # else:
    #     print(f"Failed to load image from path: {sample_image_path}. Check file existence and permissions.")

if __name__ == '__main__':
    # examine image
    img = cv.imread(IMG)
    show_close("Show Plain image", img)
    grid_pts = [(),(),(),()] # dummy
    # draw to test

    get_matrix_fill()
    

    # grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # show_close("Grayscaled", grayscale)
    # main()
