import cv2 as cv 
import numpy as np
from typing import Optional, List, Dict, Tuple
import copy
import sys

# Grid constants
GRID_WIDTH = 10
GRID_HEIGHT = 20

# HSV distance tolerance constant
COLOR_TOLERANCE = 70

# HSV reference dict for current piece detection, based on exact camera conditions
HSV_COLOR_MAP = {
    "I": [95, 160, 250], # light blue 
    "J": [110, 164, 248], # blue 
    "L": [15, 217, 240], # orange 
    "O": [26, 155, 245], # yellow 
    "S": [72, 140, 250], # green 
    "T": [134, 164, 248], # purple 
    "Z": [7, 178, 230], # red 
}

# color map dict for checking fill, based on exact camera conditions
HSV_ALL_PIECES = {
    "I": [95, 130, 250], # light blue 
    "J": [110, 164, 248], # blue 
    "L": [15, 155, 225], # orange 
    "O": [26, 155, 245], # yellow
    "S": [72, 140, 250], # green 
    "T": [134, 164, 248], # purple
    "Z": [7, 178, 230], # red 
    "GRAY": [96, 86, 255], # gray filled block 
}

def initalize_matrix_fill(img, grid_pts):
    """
    Calculate and return coordinate pixel points to check for block fill, based on input
    corners of the grid and known size. Intended to be run just once / periodically.

    Args: 
        img: opencv img numpy object representing current frame to read from.
        grid_pts: a list of four coordinates points ex. [(x1, y1), (x2, y2), ...].

    Returns:
        check_grid: a 2D array of tuples, representing the coordinate points of each grid cell's center.
    """
    # make grid of pixel coord pts to check screen image [[(x, y)]]
    check_grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    # print(check_grid)

    # start left pt, clockwise order
    for pt in grid_pts: # test draw circles of points
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
    
    # visualize on image for testing
    # show_close("Detected grid cells points", img)
    return check_grid

def check_fill(img, check_grid):
    """
    Check the color of each grid cell by pixel coordinate to see if filled or not.

    Args:
        check_grid: a 2D array of tuples, representing the coordinate points of each grid cell's center.

    Returns:
        game_state_grid: a filtered 2D array of 0s and 1s (0 empty, 1 filled) representing the game state.
        grid_p: a pre-filtered 2D array of 0s and 1s (0 empty, 1 filled) representing the game state.
    """
    # populate this grid to hold 0 and 1's, should have 200 (grid cell count)
    game_state_grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    img_copy = img.copy()

    # iterate through the cetto check the game piece
    for i, row in enumerate(check_grid): # same size
        # print(f'----- NEW DAMN ROW TO CHECK FILL -----')
        for j, cell in enumerate(row):
            # check color 
            y, x = cell
            pixel = img[y, x]
            # print(f'pixel: {pixel}')
            # print(f'START CHECKING PIXEL FILL: {i,j}')
            if classify_cell_color(pixel, HSV_ALL_PIECES):
             # access image at row pixel, col pixel
                # print("filled")
                game_state_grid[i][j] = 1

                # verify visually with a green dot
                cv.circle(img_copy, (x, y), 3, (0, 255, 255), -1) 
    
    # show_close("Check Fill", img)

    # print(f"Game grid before scrubbing: {game_state_grid}")
    # implement scrubbing of current piece
    grid = copy.deepcopy(game_state_grid) # copy for checking
    grid_p = copy.deepcopy(game_state_grid) # unscrubbed copy 

    # flip matrix entries to 0 if current piece
    for i, row in enumerate(grid): 
        for j, cell in enumerate(row):
            if cell == 1: # if land
                # print(f'RIGGHT BEFORE BFS: {game_state_grid}')
                curr_coords = bfs(i, j, grid)
                # print(f'RIGGHT AFTER BFS: {game_state_grid}')
                if curr_coords is not None: # if have detected current piece --> SCRUB
                    # print("Detected floating island of current piece!")
                    # print(f'curr_coords to scrub: {curr_coords}')
                    for coord in curr_coords:
                        # print(f"coordinate to 0 out: {game_state_grid[coord[0]][coord[1]]}")
                        x, y = coord[0], coord[1]
                        # print(f'x is: {x} and y is: {y}')
                        game_state_grid[x][y] = 0
                        # TESTING: visually draw to be blue if floating
                        draw_y, draw_x = check_grid[x][y]
                        cv.circle(img_copy, (draw_x, draw_y), 5, (255, 0, 0), -1) 
            else:
                continue

    show_close("Check Scrub Current Piece", img_copy)

    # print(f"Game state after scrub: {game_state_grid}")
    return game_state_grid, grid_p

def get_current_piece(img, coords_grid, game_state_grid):
    """
    Detect and return what the current piece is.

    Args:
        img: opencv img numpy object representing current frame to read from.
        coords_grid: a 2D array of tuples, representing the coordinate points of each grid cell's center.
        game_state_grid: the pre-filtered 2D array of 0s and 1s (0 empty, 1 filled) representing the game state.

    Returns:
        current_piece: a String representing the detected tetromino piece.
    """
    # heuristic checking top 3 rows of screen for fill
    current_piece = None
    # print(coords_grid)
    for i, row in enumerate(game_state_grid):
        for j, cell in enumerate(row):
            if i < 3:            
                if cell == 1:
                    y, x = coords_grid[i][j]
                    # print(f"Checking pixel at row {i}, col={j}")
                    pixel = img[y, x]
                    # hsv_pixel = cv.cvtColor(
                    #     np.uint8([[pixel]]),
                    #     cv.COLOR_BGR2HSV
                    # )[0][0]
                    # print(f"color = {hsv_pixel}")
                    current_piece = classify_cell_color(pixel, HSV_COLOR_MAP)
                    # print(current_piece)

    # return current_piece (either None is no detection, or classified)
    return current_piece

def classify_cell_color(bgr_color, color_map):
    """
    Find closest color in HSV_COLOR_MAP using HSV squared distance.

    Args:
        bgr_color: a NumPy array of the target pixel's BGR color values.
        color_map: a dict representing the HSV reference values for each tetromino piece.

    Returns:
        closest_name: a String representing the closest tetronimo match, or None.
    """
    # Convert BGR pixel → HSV
    hsv_pixel = cv.cvtColor(
        np.uint8([[bgr_color]]),
        cv.COLOR_BGR2HSV
    )[0][0]
    # print(f'- PIXEL bgr: {bgr_color}, hsv: {hsv_pixel}')

    min_distance = float('inf')
    closest_name = None

    # Compare to each reference HSV color using squared distance
    for name, hsv_ref in color_map.items():
        hsv_ref = np.array(hsv_ref)

        # print(f'pixel hsv: {hsv_pixel}, {name} hsv: {hsv_ref}')
        distance = hsv_distance(hsv_pixel, hsv_ref)

        if distance < min_distance:
            min_distance = distance
            closest_name = name
            # print(f'closest color piece match: {closest_name}, distance {min_distance}')

    # reject if too far
    if min_distance < COLOR_TOLERANCE ** 2:
        # print(closest_name)
        return closest_name
    else:
        return None
    
def bfs(x_og, y_og, grid):
    """
    A helper function that runs BFS to find floating island, that's not connected to bottom row.

    Args:
        x_og: an int representing the starting x coordinate of island
        y_og: an int representing the starting y coordinate of island
        grid: a 2D array of 0s and 1s (0 empty, 1 filled) representing the game state being modified.

    Returns:
        curr_coords: a list of tuples representing coordinates to be flipped to be 0 (unfilled).
    """
    queue = [(x_og, y_og)]
    visited = []

    curr_coords = []
    bottom = False

    while queue:
        node = queue.pop(0) # tuple of x, y coords (but swapped in actuality to our usual reference)

        if node not in visited:
            visited.append(node)
            x, y = node
            curr_coords.append((x,y)) # append to tentative current piece coordinate list
            if x == 19:
                bottom = True # catch starting block
            grid[x][y] = 0 # set land to checked (by switching to water)

            # check each of 4 directions + the 4 adjacent cornerss
            for dx, dy in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x - 1, y -1), (x - 1, y+1), (x+1, y+1), (x+1, y-1)]:
                # check in bounds
                if 0 <= dx < GRID_HEIGHT and 0 <= dy < GRID_WIDTH:
                    # print(f"coordinate to check: {dy, dx}") # flipped for our usual reference
                    if grid[dx][dy] == 1: # if floating island
                        queue.append((dx, dy))
                        curr_coords.append((dx,dy)) # append to tentative current piece coordinate list
                        if dx == 19:  # if connected to bottom flag of connected switch on
                            bottom = True
    if bottom is False: # if not touching bottom (floating island) --> return
        return curr_coords
    else:
        return None  

def hsv_distance(hsv1, hsv2, wH=8, wV=1):
    """
    Helper function to calculate distance between two pixel's HSV values.
    
    Excludes Saturdation, with weights given to Hue and Value.

    Args:
        hsv1: Numpy array holding HSV values of target pixel.
        hsv2: Numpy array holding HSV values of reference pixel.
        wH: int representing weight given to Hue value.
        wV: int representing weight given to Value value.
    
    Returns:
        distance: the weighted total calculation of difference between HSV values., 
    """
    dh = min(abs(hsv1[0] - hsv2[0]), 180 - abs(hsv1[0] - hsv2[0]))  # hue wrap-around
    ds = hsv1[1] - hsv2[1]
    dv = hsv1[2] - hsv2[2]
    distance = wH * dh * dh + wV * dv * dv
    return distance

def show_close(caption, img):
    """
    Helper function to simplify repeated showing and closing of cv windows.

    Args:
        caption: a string for the img caption.
        img: CV image to show.
    """
    # handle showing and close operations
    cv.imshow(caption, img) # overwrites previous img of same name if exists
    if cv.getWindowProperty(caption, cv.WND_PROP_VISIBLE) < 1:
        cv.destroyAllWindows()
        sys.exit(0)

def main(args=None):
    """
    Mock pipeline to test game state detection sequence.
    """
    # CHANGE THIS PATH to the location of your Tetris game screen
    img_path = "./assets/tetris_current_purple.jpeg" # dummy image path

    # Load, run detect once.
    frame_to_process = cv.imread(img_path)
    if frame_to_process is not None:
        # print(f"Successfully loaded image: {img_path}. Running detection...")
        # show_close("Show Plain image", frame_to_process)
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
