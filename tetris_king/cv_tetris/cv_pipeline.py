import game_state_detection
import grid_detection
import cv2 as cv 

def initialize_video_capture():
    """
    Returns cv VideoCapture object, None is video source failed
    """
     # feed in video 
    cap = cv.VideoCapture(0)

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
    Returns cv img frame to read from

    frame: img numpy frame
    """
    # read frame
    ret,frame = cap.read()

    if ret:
    # Display the first frame using imshow
        cv.imshow("Captured Frame", frame)
        cv.waitKey(0)  # Wait for a key press to close the window
        cv.destroyAllWindows()  # Close the window
        return ret, frame
    else:
        print("Error: Could not read a frame from the sequence.")
        return None

def initialize_grid(img):
    """
    Args:
    - img: 
    """
    pts = grid_detection.get_grid_coords(img)
    # print(pts)
    check_grid = game_state_detection.initalize_matrix_fill(img, pts)
    return check_grid

def get_cv_info(cap):
    """
    Final function called by game super loop that returns game_state matrix and current_piece

    Args:
    - cap: CV VideoCapture object

    Returns:
    - tuple of game_state, current_piece
    """
    game_state = None
    current_piece = None
    ret, img = video_to_frame(cap) 
    if ret: # if got frame correctly, check fill
        game_state, current_piece = process_image(img)
    print(game_state, current_piece)
    return game_state, current_piece

def process_image(img):
    print("Got frame")
    # get grid_pts
    grid_img = img.copy()
    grid_pts = initialize_grid(grid_img) 
    # get game_state matrix
    game_state = game_state_detection.check_fill(img, grid_pts)
    # get current piece
    current_piece = game_state_detection.get_current_piece(img, grid_pts, game_state)
    # print(current_piece)
    return game_state, current_piece

def test_frame():
    """
    Mock game system pipeline with image path, abstracting away model + arm.
    """
    img_path = "./assets/versus.jpg" # dummy image path for now

    img = cv.imread(img_path)

    game_state = None
    current_piece = None
    game_state, current_piece = process_image(img)
    return game_state, current_piece

if __name__ == "__main__":
    test_frame()
    # cap = initialize_video_capture()
    # game_state, current_piece = get_cv_info(cap)
