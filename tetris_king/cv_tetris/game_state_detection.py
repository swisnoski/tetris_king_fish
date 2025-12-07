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
    # "EMPTY": [0, 0, 0],         # Background
    "I": [245, 106, 0],     # Dark
    "J": [240, 60, 0],    # Dark Blue
    "L": [85, 96, 226],   # Orange
    "O": [120, 160, 170],   # Yellow
    "S": [155, 200, 0],     # Green
    "T": [209, 19, 74],   # Purple
    "Z": [61, 17, 202],     # Red
    "GRAY": [215, 215, 215]
}

GRAY_BLOCK = [215, 215, 215] # for the inserted puyo puyo fill blocks

offset_cell_x = 0 # deal with code scoping later

def initalize_matrix_fill(img, grid_pts):
    """
    Calculates and returns coordinate pixel points to check for block fill, based on input
    corners of the grid and known size. Intended to be run just once / periodically.

    Args: 
        - a list of four coordinates points ex. [(x1, y1), (x2, y2), ...],

    Returns:
        - check_grid: a 2D array of tuples, representing the coordinate pixel points of the center of each grid cell
    """
    # ----------
     # TO CONSIDER --> THIS LOWK JUST NEEDS TO RUN ONCE (JUST GRID?)
    # make grid of pixel coord pts to check screen image [[(x, y)]]
    # check_grid = None
    check_grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    # print(check_grid)

    # use dummy data first -- assume start left pt, clockwise order
    # test draw circles of points
    for pt in grid_pts:
        cv.circle(img, pt, 5, (255, 255, 255), -1) 
    left_x = grid_pts[0][0]
    right_x = grid_pts[1][0] 
    up_y = grid_pts[0][1]
    bottom_y = grid_pts[2][1]
    # calculate the pixel center points to fill grid 
    width = right_x - left_x # upper right x - upper left x
    height = bottom_y - up_y   # bottom left y - upper left y 

    cell_len_x = width / GRID_WIDTH # width / grid_num (10)
    cell_len_y = height / GRID_HEIGHT # height / grid_num (20)
    offset_cell_x = cell_len_x / 2 # half of cell to offset to center
    offset_cell_y = cell_len_y / 2 # half of cell to offset to center

    # fill matrix with coords of each cell's center point 
    for i in range(GRID_HEIGHT): # go through 20 
        for j in range(GRID_WIDTH): # go through row
            # print(f'i: {i}, j: {j}')
            # set coordinate point
            y, x = up_y + (i * cell_len_y + offset_cell_y), left_x + (j * cell_len_x + offset_cell_x) 
            y, x = int(y), int(x)
            check_grid[i][j] = y, x
            # print((x, y))
            # verify for now with drawing point on image
            cv.circle(img, (x, y), 5, (255, 255, 255), -1) 
    show_close("Detected grid cells points", img)
    return check_grid

def check_fill(img, check_grid):
    """
    Checks the color of each grid cell by pixel coordinate to see if filled

    Args:
    - check_grid: a 2D array of tuples, representing the coordinate pixel points of the center of each grid cell

    Returns
     - output_grid: a 2D array of 0s and 1s (0 empty, 1 filled) ex. [[0, 0, 0],[0, 1, 0]], 2 is ghost
    """
    # populate this grid to hold 0 and 1's, should have 200 (grid cell count)
    game_state_grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

    # iterate through the cetto check the game piece
    for i, row in enumerate(check_grid): # same size
        for j, cell in enumerate(row):
            # check color 
            y, x = cell
            pixel = img[y, x]
            # print(f'pixel: {pixel}')
            # print("check fill")
            if classify_cell_color(pixel):
             # access image at row pixel, col pixel
                print("filled")
                game_state_grid[i][j] = 1
                # verify visually with a green dot
                cv.circle(img, (x, y), 5, (0, 255, 0), -1) 
            # else: # from not filled pieces, check for ghost piece (current piece's outline)
            #     # check both 2 points piece outward off from center (L & R)
            #     ghost_offset = offset_cell_x * 0.6 # lol 50 * 0.6 for 30%?
            #     left_pixel = img[y, int(x - ghost_offset)]
            #     hsv_left = cv.cvtColor(np.uint8([[left_pixel]]), cv.COLOR_BGR2HSV)[0][0]
            #     right_pixel = img[y, int(x + ghost_offset)]
            #     hsv_right = cv.cvtColor(np.uint8([[right_pixel]]), cv.COLOR_BGR2HSV)[0][0]
            #     hue_threshold = 286 / 2 # 286 regular divided by 2 for opencv scale; 145
            #     # print(hsv_left[0], hsv_right[0])
            #     if hsv_left[0] > hue_threshold and hsv_right[0] > hue_threshold:
            #         game_state_grid[i][j] = 2 # then ghost piece
            #         cv.circle(img, (x, y), 5, (255, 0, 0), -1) # check with bllue overlay
    
    show_close("Check Fill", img)

    # implement scrubbing of current piece
    
    # print(output_grid)
    return game_state_grid

def get_player_pieces():
    """
    Returns a list containing current and next pieces

    Returns:
    - a list in order of current -> next hold pieces
    """

def get_current_piece(img, coords_grid, game_state_grid):
    """
    Detect and return what the current piece is
    """
    # consider not recalculating if the piece is the same as the previously detected piece
    # need some saved dict of pieces here 
    # should determine this based on the glowing outline at screen bottom
        # --> accounts for when the current piece may not be on the screen yet 
        # --> also matches how actual players see the current piece first
    # mvp clunky heuristic checking top of screen with 2 block gap
    current_piece = None
    for i, row in enumerate(game_state_grid):
        for j, cell in enumerate(row):
            if i < 3:            
                if cell == 1:
                    pixel = img[coords_grid[i][j]]
                    # print(f'pixel = {pixel}')
                    current_piece = classify_cell_color(pixel)
                    # print(result)

    # return current_piece (either None is no detection, or classified)
    return current_piece
    # if no piece detected, default to last_current
    # if result is None: 
    #     current_piece = last_current
    # elif result != last_current: # if different as last current piece
    #     current_piece = result # check this logic 
    # else: # if result is same as last_current
    #     current_piece = last_current
    # return current_piece

# Color tolerance (higher=easier match)
COLOR_TOLERANCE = 65
# Squared for distance check.

def classify_cell_color(bgr_color: np.ndarray) -> str:
    """Finds closest color in COLOR_MAP."""
    min_distance = float('inf')
    closest_color_name = "EMPTY"
    
    for name, color in COLOR_MAP.items():
        color_array = np.array(color)
        
        # Squared distance (B,G,R space).
        distance = np.sum((bgr_color.astype(int) - color_array.astype(int)) ** 2)

        # print(distance)
        
        if distance < min_distance:
            min_distance = distance
            closest_color_name = name
    
    # Check if distance OK.
    if min_distance < COLOR_TOLERANCE ** 2:
        print(closest_color_name)
        return closest_color_name
    else:
        return None
        
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
    # CHANGE THIS PATH to the location of your Tetris game screen
    img_path = "./assets/tetris_current_purple.jpeg" # dummy image path for now
    # img_path = "./assets/tetris_screen_cleaned.jpeg"

    # Load, run detect once.
    frame_to_process = cv.imread(img_path)
    if frame_to_process is not None:
        print(f"Successfully loaded image: {img_path}. Running detection...")
        show_close("Show Plain image", frame_to_process)
        # grid_pts = [(420, 250), (1000, 250), (1000, 1400), (420, 1400)] # dummy for start_tetris_cleaned.jpeg
        # grid_pts = [(255, 150), (810, 150), (810, 1290), (255, 1290)] # dummy for tetris_screen_clean.jpeg
        grid_pts = [(240, 150), (800, 150), (800, 1280), (255, 1280)] # dummy for tetris_current_purple.jpeg
        draw_verification_img = cv.imread(img_path) # copy of image for visualization testing
        check_grid = initalize_matrix_fill(draw_verification_img, grid_pts)
        game_state_img = cv.imread(img_path) # copy of image for visualization testing
        game_state_grid = check_fill(game_state_img, check_grid)
        
        # get current piece
        get_current_piece(frame_to_process, check_grid, game_state_grid)
    else:
        print(f"Failed to load image from path: {img_path}. Check file existence and permissions.")

if __name__ == '__main__':
    main()
