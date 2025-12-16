from . import game_state_detection
from . import grid_detection
# import game_state_detection
# import grid_detection
import cv2 as cv 

def initialize_video_capture():
    """
    Returns cv VideoCapture object, None is video source failed
    """
    # feed in video 
    cap = cv.VideoCapture(0)
    # UNCOMMENT if not live feed:
    # cap = cv.VideoCapture("./assets/video_final.mp4")        

    # check for error:
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return None
        # exit()
    else:
        print("Video capture opened successfully")
    
    return cap

def video_to_frame(cap):
    """
    Helper function to return cv img frame to read from

    frame: img numpy frame
    """
    # UNCOMMENT FOR RECORDED VID TESTING ONLY: skips to go 5 frames at a time
    # for _ in range(20):
    #     cap.grab()   # fast skip

    # read frame
    ret,frame = cap.read()

    if ret:
    # Display the first frame using imshow
        cv.imshow("Captured Frame", frame)
        return ret, frame
    else:
        print("Error: Could not read a frame from the sequence.")
        return None
    
def initialize_grid(cap):
    """
    Initialize grid corners and cells coordinates, waits for approval of user hitting 'g'
    """
    corner_pts = None
    grid_pts = None
    grid_ready = False

    # loop until approval for initializing grid
    while grid_ready is False:
        ret, img = video_to_frame(cap) 
        # get corner_pts from frame, loop until actually detecting a grid
        while corner_pts is None:
            if ret: # if got frame correctly, check fill
                print("Got frame")
                # get grid_pts
                # corner_pts = [(220, 40),(470, 40),(470, 500),(220, 500)]
                corner_pts = grid_detection.get_grid_coords(img)
            if corner_pts is None:
                ret, img = video_to_frame(cap) 
            # print(f'Corner points: {corner_pts}')

        # get coordinates of cells, hang until 'g' pressed to continue
        grid_img = img.copy()
        grid_pts = game_state_detection.initalize_matrix_fill(grid_img, corner_pts)

        # visualize grid initalization
        cv.putText(grid_img, "Press g if ready, any other key to re-try grid search", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        cv.imshow('Grid Initialization', grid_img)
        if cv.waitKey(0) == ord('g'):
            grid_ready = True
            cv.destroyWindow('Grid Initialization')
        else:
            # reset corner and grid points
            corner_pts = None
            grid_pts = None

    return grid_pts
    
def get_cv_info(cap, grid_pts):
    """
    Function called by game super loop that returns game_state matrix and current_piece

    Args:
    - cap: CV VideoCapture object

    Returns:
    - tuple of game_state, current_piece
    """
    game_state = None
    current_piece = None
    
    ret, img = video_to_frame(cap) 
    if ret:
        # detect filled cells
        game_state, game_state_p = game_state_detection.check_fill(img, grid_pts)
        # get current piece
        current_piece = game_state_detection.get_current_piece(img, grid_pts, game_state_p)
        # print(f"current p: {current_piece}")
        # print(current_piece)
        # print(f'Final game state and current piece: {game_state, current_piece}')
        return game_state, current_piece
    else:
        print("Error: Could not read a frame from the sequence.")
        return None, None
    
# --------------------------- HELPER FUNCTIONS FOR TESTING -----------------------------------
    
def get_grid(img):
    """
    Helper function 
    Args:
    - img: opencv img numpy obj ect 
    """
    pts = None
    # while pts is None:
    # pts = grid_detection.get_grid_coords(img)
    # if pts is None:
    #     return
    # print(pts)
    # pts = [(100, 70),(280, 70),(280, 425),(100, 425)] # DUMMY POINTS FOR GRID VIDEO video_test.mp4
    # pts = [(250, 150),(825, 150),(825, 1300),(250, 1300)] # DUMMY POINTS FOR GRID VIDEO
    # pts = [(250, 150),(825, 150),(825, 1300),(250, 1300)] # DUMMY POINTS FOR GRID VIDEO video_final.mp4
    pts = [(190, 60),(640, 60),(640, 980),(190, 980)] # DUMMY for tetris 
    check_grid = game_state_detection.initalize_matrix_fill(img, pts)
    return check_grid

def process_image(img):
    """
    Helper function for all things just needing an img, with game_state and fill detection
    """
    # print("Got frame")
    # get grid_pts
    grid_img = img.copy()
    grid_pts = initialize_grid(grid_img)
    game_state = game_state_detection.check_fill(img, grid_pts)
    # get current piece
    current_piece = game_state_detection.get_current_piece(img, grid_pts, game_state)
    # print(current_piece)
    return game_state, current_piece

# --------------------------- TESTING FUNCTIONS -----------------------------------

def test_frame():
    """
    Mock game system pipeline with image path, abstracting away model + arm.
    """
    img_path = "./assets/tetris_final.png" # test image path

    img = cv.imread(img_path)

    game_state = None
    current_piece = None
    game_state, current_piece = process_image(img)
    return game_state, current_piece

def test_video_once():
    cap = initialize_video_capture()
    game_state, current_piece = get_cv_info(cap)

def test_video_loop():
    cap = initialize_video_capture()
    try:
        while True:
            game_state, current_piece = get_cv_info(cap)

            # allow KeyboardInterrupt
            if cv.waitKey(20) == ord('q'):
                break
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        cap.release()

def test_new_video_loop():
    cap = initialize_video_capture()
    try:
        grid_pts = initialize_grid(cap)
        while True:
            game_state, current_piece = get_cv_info(cap, grid_pts)
            # allow KeyboardInterrupt
            if cv.waitKey(20) == ord('q'):
                break
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        cap.release()

if __name__ == "__main__":
    # test_frame()
    test_new_video_loop()
    
