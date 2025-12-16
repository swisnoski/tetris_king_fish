import cv2 as cv 
import numpy as np
from typing import Optional, List, Dict, Tuple
import copy

# --- Grid constants ---
GRID_WIDTH = 10
GRID_HEIGHT = 20

HSV_COLOR_MAP = {
    "I": [95, 250, 250], # light blue [95 144 252] [95 145 252] [97 145 251] [ 96 142 252] slight outlier: [ 90 122 254] [ 90 210 255] [ 90 217 255]
    "J": [110, 164, 248], # blue [111 177 247] [110 190 247] [109 158 248] [109 139 248]
    "L": [15, 155, 225], # orange [15 162 226] [14 150 224]
    "O": [35, 90, 245], # yellow [35, 88, 240] [35, 88, 240] [32, 98, 252] [32, 94, 255]
    "S": [150, 255, 250], # green [70 138 214]; [ 74 228 255]
    "T": [134, 164, 248], # purple 19,0: [133 158 247] [135 134 248]; [139 183 255], [135, 151, 255], [132, 196, 255]
    "Z": [7, 178, 230], # red [6 174 231] [8 180 231]
}

HSV_ALL_PIECES = {
    "I": [95, 250, 250], # light blue [95 144 252] [95 145 252] [97 145 251] [ 96 142 252] slight outlier: [ 90 122 254] [ 90 210 255] [ 90 217 255]
    "J": [110, 164, 248], # blue [111 177 247] [110 190 247] [109 158 248] [109 139 248]
    "L": [15, 155, 225], # orange [15 162 226] [14 150 224]
    "O": [35, 90, 245], # yellow [35, 88, 240] [35, 88, 240] [32, 98, 252] [32, 94, 255]
    "S": [150, 255, 250], # green [70 138 214]; [ 74 228 255]
    "T": [134, 164, 248], # purple 19,0: [133 158 247] [135 134 248]; [139 183 255], [135, 151, 255], [132, 196, 255]
    "Z": [7, 178, 230], # red [6 174 231] [8 180 231]
    "GRAY": [96, 86, 255], # gray puyo block [95, 35, 100]
}

# Color tolerance (higher=easier match)
COLOR_TOLERANCE = 70

def initalize_matrix_fill(img, grid_pts):
    """
    Calculates and returns coordinate pixel points to check for block fill, based on input
    corners of the grid and known size. Intended to be run just once / periodically.

    Args: 
        - a list of four coordinates points ex. [(x1, y1), (x2, y2), ...],

    Returns:
        - check_grid: a 2D array of tuples, representing the coordinate pixel points of the center of each grid cell
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
    Checks the color of each grid cell by pixel coordinate to see if filled

    Args:
    - check_grid: a 2D array of tuples, representing the coordinate pixel points of the center of each grid cell

    Returns
     - output_grid: a 2D array of 0s and 1s (0 empty, 1 filled) ex. [[0, 0, 0],[0, 1, 0]], 2 is ghost
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
    
    # cv.namedWindow("Preview", cv.WINDOW_NORMAL)
    # show_close("Check Fill", img)

    # print(f"Game grid before scrubbing: {game_state_grid}")



    # implement scrubbing of current piece
    grid = copy.deepcopy(game_state_grid) # copy for checking
    grid_p = copy.deepcopy(game_state_grid) # copy for checking

    def bfs(x_og, y_og):
        """
        A helper function that runs BFS to find floating island, that's not connected to bottom row

        Args:
            x_og: an int representing the starting x coordinate of island
            y_og: an int representing the starting y coordinate of island
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
                    if 0 <= dx < len(grid) and 0 <= dy < len(row):
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
    
    # flip matrix entries to 0 if current piece
    for i, row in enumerate(grid): 
        for j, cell in enumerate(row):
            if cell == 1: # if land
                # print(f'RIGGHT BEFORE BFS: {game_state_grid}')
                curr_coords = bfs(i, j)
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
    Detect and return what the current piece is
    """
    # heuristic checking top of screen with 2 block gap
    current_piece = None
    # print(coords_grid)
    for i, row in enumerate(game_state_grid):
        for j, cell in enumerate(row):
            if i < 5:            
                if cell == 1:
                    x, y = coords_grid[i][j]
                    # print(f"Checking pixel at x={x}, y={y}")
                    pixel = img[y, x]
                    # print(f"pixel = {pixel}")
                    # show_close("img", img)
                    # Check a small 3x3 area to see the actual colors
                    # region = img[y-1:y+2, x-1:x+2]
                    # print("Region:\n", region)
                    current_piece = classify_cell_color(pixel, HSV_COLOR_MAP)
                    # print(current_piece)

    # return current_piece (either None is no detection, or classified)
    return current_piece

def classify_cell_color(bgr_color, color_map) -> str:
    """
    Finds closest color in HSV_COLOR_MAP using HSV squared distance.
    Returns color name or None.
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

def hsv_distance(hsv1, hsv2, wH=8, wS=1, wV=1):
    dh = min(abs(hsv1[0] - hsv2[0]), 180 - abs(hsv1[0] - hsv2[0]))  # hue wrap-around
    ds = hsv1[1] - hsv2[1]
    dv = hsv1[2] - hsv2[2]
    return wH * dh * dh + wS * ds * ds + wV * dv * dv

def show_close(caption, img):
    """
    Helper function to simplify repeated showing and closing of cv windows

    caption: a string for the img caption
    img: CV image to show
    """
    # handle showing and close operations
    cv.imshow(caption, img) # overwrites previous img of same name if exists
    if cv.getWindowProperty(caption, cv.WND_PROP_VISIBLE) < 1:
        import sys
        cv.destroyAllWindows()
        sys.exit(0)

def main(args=None):
    # CHANGE THIS PATH to the location of your Tetris game screen
    img_path = "./assets/tetris_current_purple.jpeg" # dummy image path for now
    # img_path = "./assets/tetris_screen_cleaned.jpeg"

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
