## System Architecture 

Our system runs a superloop of three main components in this order: computer vision, a custom tetris emulator and algorithm, and a robot arm.
<p align="center">
  <img src="./system_architecture.png" alt="System Architecture" width="70%">
  <br>
  <em>Figure 1: System Pipeline</em>
</p>

**Computer Vision**

The computer vision pipeline runs in this sequence:
- Grid detection (*grid_detection.py*)
- Fill detection of cells (*game_state_detection.py*)
- Current piece detection (*game_state_detection.py*)

Our grid detection takes in a frame of the camera feed, and detects the gameplay grid of the Tetris blocks, outputting the four coordinates points of the grid corners. 

These four coordinate points are then used by the fill detection in order to calculate the center of each grid cell for later color mapping. We calculate the width and height of the gameplay grid using the detected four coordinate points, using the known dimensions of the gameplay grid in cells (10 by 20 cells always) to divide the width and height in cells counts, and applying an offset of half the length in each direction to find the center of the grid. 

<p align="center">
  <img src="./grid_initialization.png" alt="Detected Grid" width="20%">
  <br>
  <em>Figure 2: Our detected grid corner points with center cell points</em>
</p>

From these center coordinates, we determinate by using a HSV color map for each tetris piece to check whether a cell is filled, by checking the HSV pixel for being within distance of a certain threshold of any HSV value of a tetris piece. If the cell center pixel passes this threshold, we consider it "filled", and otherwise empty. These get recorded in an output of a 2D array representing the game grid in 0 and 1s. If a cell is 0, then it is empty, and if it is 1, then the cell is detected as filled. 

<p align="center">
  <img src="./cv_check_fill.png" alt="CV Detected Filled Cells" width="20%">
  <br>
  <em>Figure 3: Filled cells detected in green dots</em>
</p>

Since we don't want our current falling piece to be considered as part of the "filled" section of the game state, we also exclude the current piece from this fill detection by using a breadth-first search algorithm to determine whether there is an "island" of detected filled pieces that isn't connected to the bottom of the grid, and excluding it. Figure 4 shows the detected pieces to scrub from its consideration of a "filled" game state:

<p align="center">
  <img src="./scrub_piece.png" alt="CV Scrub Piece" width="20%">
  <br>
  <em>Figure 4: The detected pieces dotted in blue to scrub from its consideration of a "filled" game state</em>
</p>

The current piece detection is accomplished by checking the upper three rows of the gameplay board for a piece using a similar color detection to the fill detection, as we know a piece always spawns in the same place.

To integrate smoothly with the other components of the project, we wrapped these processes into two functions that the overall pipeline for the project uses, from *cv_pipeline.py*: 
- `initialize_video_capture() `
- `initialize_grid()`
- `get_cv_info()`

The function `initialize_video_capture()` initializes the webcam video capture using OpenCV's built in `VideoCapture()` function, and runs once at our program start. `initialize_grid()` intitializes the grid corner points and the center of the game cells' coordinates, allowing users to look at the detection visually and re-try until the grid is locked on accurately. This can be seen below:

<p align="center">
  <img src="./cv_beginning_grid.gif" alt="CV Grid Initialization" width="50%">
  <br>
  <em>Figure 5: Grid Initialization Process</em>
</p>

From there, `get_cv_info()` is called at the beginning of every program loop, in order update the game state and current piece. These repeated loops and CV game state detction can be seen below, with testing on a recorded video of gameplay.

<p align="center">
  <img src="./cv_gamestate.gif" alt="CV Gamestate Loop" width="50%">
  <br>
  <em>Figure 6: Game state detection loop, with yellow dots indicating detected filled cells and blue dots representing filled cells to scrub out</em>
</p>

## Design Decisions

**Computer Vision**

One major design decision we made early on was to cut out the first step of screen detection, in order to reduce unneeded complexity with our image input. Instead, we position our webcam to largely only take in the player area we want to control.

<p align="center">
  <img src="../tetris_king/cv_tetris/gamescreen_test_assets/tetris_screen_final.png" alt="Input Frame for CV" width="40%">
  <br>
  <em>Input Frame for CV</em>
</p>

