import cv2 as cv 
import numpy as np

# ----------- Detecting TV screen -----------
def detect_TV():
    """
    Detect TV Screen from initial Tetris camera image, and crop image
    to just the screen contents.
    """
    # load in image
    img = cv.imread("./assets/start_tetris.jpg")

    # resize image so not massive
    original_height, original_width = img.shape[:2]
    new_width = 750
    aspect_ratio = new_width / original_width # get new aspect ratio
    new_height = int(original_height * aspect_ratio)
    img = cv.resize(img, (new_width, new_height))

    # grayscale 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    show_close("Grayscaled", gray)

    # blur image
    blur = cv.bilateralFilter(gray, 9, 75, 75)
    show_close("Blurred", blur)

    # a color mask? since the black is so prominent 
    # ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    # skin_mask = cv.inRange(ycrcb, (0,133,77), (255,173,127))
    # blur[skin_mask > 0] = 0
    # show_close("Color Mask", )

    # detect edges
    edges = cv.Canny(blur, 100, 200)
    show_close("Edges Detected", edges)

    # morphological closing

    # contour detection

# ------- Detecting gameplay area -----------


# ------ Initalizing and masking the grid --------
def detect_grid():
    pass

def show_close(caption, img):
    """
    Helper function to simplify repeated showing and closing of cv windows

    caption: a string for the img caption
    img: CV image to show
    """
    cv.imshow(caption, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    # main()
    detect_TV()
