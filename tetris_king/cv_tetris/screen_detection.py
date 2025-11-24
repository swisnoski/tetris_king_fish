import rclpy
from rclpy.node import Node
import cv as cv 
import numpy as np
from typing import Optional, List, Dict, Tuple

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


class TetrisGridDetector(Node):
    """ROS 2 node. Finds grid, fixes angle, reads state."""

    def __init__(self):
        super().__init__('tetris_grid_detector')
        self.get_logger().info('Tetris Grid Detector Node initialized. Ready for image processing.')

        # --- Grid detect thresholds ---
        # TUNE BGR to find grid lines.
        # Used in cv.inRange.
        self.LOWER_THRESHOLD = np.array([200, 200, 200]) # BGR light colors (placeholder)
        self.UPPER_THRESHOLD = np.array([255, 255, 255]) # Max BGR (placeholder)
        
        # Color tolerance (higher=easier match)
        self.COLOR_TOLERANCE = 30 
        # Squared for distance check.


    def classify_cell_color(self, bgr_color: np.ndarray) -> str:
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
        if closest_color_name != "EMPTY" and min_distance < self.COLOR_TOLERANCE ** 2:
            return closest_color_name
        else:
            return "EMPTY"
            
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """Orders 4 grid corners (TL, TR, BR, BL). Essential for warp."""
        rect = np.zeros((4, 2), dtype="float32")
        
        # Calc sums/diffs.
        s = pts.sum(axis=1)
        # TL=min sum, BR=max sum.
        rect[0] = pts[np.argmin(s)]  
        rect[2] = pts[np.argmax(s)]  
        
        # TR=min diff, BL=max diff.
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] 
        rect[3] = pts[np.argmax(diff)] 
        
        return rect


    def process_image(self, frame: np.ndarray) -> Optional[List[List[str]]]:
        """Find grid, fix angle, read state."""
        try:
            if frame is None:
                self.get_logger().error("Input frame is None.")
                return None
            
            # Make a copy for drawing the boundary
            display_frame = frame.copy() 

            # 1. Find grid boundary (color/contour)
            # Mask grid lines.
            mask = cv.inRange(frame, self.LOWER_THRESHOLD, self.UPPER_THRESHOLD)
            # Canny edges.
            edges = cv.Canny(mask, 50, 150, apertureSize=3)
            # Find shapes.
            contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            if not contours:
                self.get_logger().warn("No significant grid contours found.")
                # Show original image even if detection failed
                cv.imshow("Original Image - Boundary Not Found", display_frame) 
                cv.waitKey(0)
                return None

            # Largest contour = full grid.
            largest_contour = max(contours, key=cv.contourArea)
            # Approx polygon (needs 4 sides).
            epsilon = 0.04 * cv.arcLength(largest_contour, True)
            approx = cv.approxPolyDP(largest_contour, epsilon, True)

            # Check if large 4-sided shape.
            if len(approx) != 4 or cv.contourArea(largest_contour) < 1000:
                self.get_logger().warn("Could not find a clear 4-sided rectangular grid boundary.")
                # Show original image even if detection failed
                cv.imshow("Original Image - Boundary Not Found", display_frame) 
                cv.waitKey(0)
                return None
            
            # Draw the detected boundary on the original image (Green line, 3px thick)
            cv.polylines(display_frame, [approx], True, (0, 255, 0), 3)
            cv.imshow("Original Image - Detected Boundary", display_frame)
            
            # --- Perspective Warp (Deskew) ---
            
            # 2. Order 4 corners.
            rect_points = self.order_points(approx.reshape(4, 2))

            # 3. Define perfect rectangle dest.
            dst = np.array([
                [0, 0],                                 # Top-Left
                [WARPED_WIDTH - 1, 0],                  # Top-Right
                [WARPED_WIDTH - 1, WARPED_HEIGHT - 1],  # Bottom-Right
                [0, WARPED_HEIGHT - 1]], dtype="float32") # Bottom-Left

            # 4. Get warp matrix, apply warp.
            M = cv.getPerspectiveTransform(rect_points, dst)
            # 'warped_grid' is straight view.
            warped_grid = cv.warpPerspective(frame, M, (WARPED_WIDTH, WARPED_HEIGHT))
            self.get_logger().info(f"Perspective corrected grid size: {warped_grid.shape[:2]}")

            # 5. Loop cells, read state.
            cell_w = WARPED_WIDTH // GRID_WIDTH
            cell_h = WARPED_HEIGHT // GRID_HEIGHT
            
            # Draw the internal grid lines on the warped image (White lines, 1px thick)
            for i in range(1, GRID_WIDTH):
                x = i * cell_w
                cv.line(warped_grid, (x, 0), (x, WARPED_HEIGHT), (255, 255, 255), 1)
            for i in range(1, GRID_HEIGHT):
                y = i * cell_h
                cv.line(warped_grid, (0, y), (WARPED_WIDTH, y), (255, 255, 255), 1)
            
            # Init 20x10 state array.
            game_state: List[List[str]] = [["EMPTY"] * GRID_WIDTH for _ in range(GRID_HEIGHT)]

            for row in range(GRID_HEIGHT):
                for col in range(GRID_WIDTH):
                    # Center coords in warped image.
                    center_x = col * cell_w + cell_w // 2
                    center_y = row * cell_h + cell_h // 2
                    
                    # Sample center color (BGR).
                    bgr_color = warped_grid[center_y, center_x]
                    
                    piece_type = self.classify_cell_color(bgr_color)
                    game_state[row][col] = piece_type
                    
                    # (Optional) Draw dots for visual check.
                    if piece_type != "EMPTY":
                        # White dots on pieces.
                        cv.circle(warped_grid, (center_x, center_y), 5, (255, 255, 255), -1) 

            # 6. Log result.
            self.get_logger().info("--- Current Game State Read (20 rows x 10 cols) ---")
            for row_index, row_data in enumerate(game_state):
                # Log simplified board (I/./etc).
                simplified_row = "".join([s[0] if s != "EMPTY" else "." for s in row_data])
                self.get_logger().info(f"Row {row_index:02}: {simplified_row}")

            # Show the perspective corrected image with the drawn grid.
            cv.imshow("Corrected Grid State with Cell Lines", warped_grid)
            
            # Wait indefinitely until a key is pressed to keep the windows open.
            cv.waitKey(0) 
            cv.destroyAllWindows()
            
            return game_state

        except Exception as e:
            self.get_logger().error(f"An error occurred during game state reading: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)
    detector_node = TetrisGridDetector()
    
    # --- TESTING LOGIC ---
    # 1. CHANGE THIS PATH to the location of your Tetris screenshot/photo!
    sample_image_path = 'Tetris_game_ss.png'
    
    if sample_image_path == 'sample_image_path':
        detector_node.get_logger().warn(
            "ACTION REQUIRED: Please update 'sample_image_path' in main() to your actual image path for testing."
        )

    else:
        # Load, run detect once.
        frame_to_process = cv.imread(sample_image_path)
        if frame_to_process is not None:
            detector_node.get_logger().info(f"Successfully loaded image: {sample_image_path}. Running detection...")
            detector_node.process_image(frame_to_process)
        else:
            detector_node.get_logger().error(f"Failed to load image from path: {sample_image_path}. Check file existence and permissions.")

    # Cleanup ROS 2.
    detector_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
