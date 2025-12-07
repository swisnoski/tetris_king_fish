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

def initialize_grid(img):
    """
    Args:
    - img: 
    """
    pts = grid_detection.get_grid_coords(img)
    # print(pts)
    check_grid = game_state_detection.initalize_matrix_fill(img, pts)
    return check_grid


def get_cv_info():
    """
    Final function called by game super loop that returns game_state matrix and current_piece
    """
    pass

def main():
    """
    Mock game system pipeline with video feed, abstracting away model + arm.
    """
    # initalize video
    initialized = False
    cap = initialize_video_capture()
    if cap: # get initial frame to process
        ret, img = video_to_frame(cap) 

        if ret:
            # later once: initalize grid ONCE
            # print(f'img before initial: {img}')
            grid_pts = initialize_grid(img)
            print("Initalized Grid")
            initialized = True

    # try:
    while initialized:
        try:
            print("In loop now")
            current_piece = None
            ret, img = video_to_frame(cap) 
            if ret: # if got frame, check fill
                grid_pts = initialize_grid(img) # get grid_pts
                game_grid = game_state_detection.check_fill(img, grid_pts)
                current_piece = game_state_detection.get_current_piece(img, grid_pts, game_grid, current_piece)
                print(current_piece)
        except KeyboardInterrupt:
            print("Loop ended")

    # loop:
    # # read in game state -> send out flag for if different (piece placed)
    # game_state_detection.check_fill()
    # # if game_updated:
    #     # do stuff
    # # read in current piece -> update only if different
    # current_piece = game_state_detection.get_current_piece()
    # while True:
    #     img = video_to_frame()
    
if __name__ == "__main__":
    main()
