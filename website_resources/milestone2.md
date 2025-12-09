# Milestone 2 Progress Update



## Component Progress 

**1. Computer Vision:**


**2. Tetris Emulator:**


Our tetris emulator works! At this point, we’ve created three different versions of our tetris class: a playable emulator, which was mainly for debugging the original game, a model that uses our custom heuristic as a basic standard, and a model that interfaces as an RL training environment with a custom step function. 

Here is a visualization of the tetris heuristic max:   

<p align="center">
  <img src="tetris_max.gif" alt="tetris heuristic" width="200">
</p>

First, let’s explore how our main tetris class works: 

The original tetris functions by manipulating a 10x20 matrix that begins as zeros. There is a border surrounding this matrix, and we use a simple detect collision function each time we move our piece on the board. If the piece collides with a side, then we simply stop its movement. When a piece hits the bottom, we mark that it’s reached its final position, update the board to have ones instead of zeros where the piece was, and then spawn our next piece. (Actually, before we spawn our next piece, we check if we need to clear lines or if we have lost). With all of that implemented, and the fact that our new piece can now collide into our placed pieces, we can play tetris. 

Next, each of our different versions of tetris controls the movement/manipulation of pieces in different ways.
For the original model, we simply record keyboard input and move the piece left, right, or rotate the piece based on what the user inputs.
For our heuristic model, we’ve created a function that can score a board based on four factors: bumpiness, height, holes, and lines cleared. With each piece we place, we calculate every possible move for that piece and then place it in the position where we score the highest. 
Lastly, our RL model has a custom step function that takes in an action and returns the ‘next_state’, ‘reward’, ‘win_variable’, ‘iteration’, and ‘valid_moves’. Additionally, we implemented a simple reset function.  


Largely, the work for this section is completed. While there is slightly more work to be done as we move towards integrating the final model, the simulator itself is finalized. 



**3. Reinforcement Learning:**


**4. Robotic Arm Controls:** 

For this milestone, the objective is to optimize the arm speed to enable for realtime interfacing with a physical switch controller. Before the milestone, the average time for the arm to press a button and move back to the "home" position was 3s. Now, that time has decreased to 1.5s. We are still looking to decrease it even further if possible.

Here are the specific tasks achieved in this milestone:
- Physical setup with arm and controller, currently being held in place by wall putty.
- Integrating arm ROS node with `/action` topic which publishes list: `[rotations, moves left/right]`. Arm node receives data and performs required movement to press buttons on the controller
- Implement threading to increase task speed. Works 90% of the time. However, at times, the arm will duplicate one task, most likely a concurrency problem.

Goals til end of project:
- Fix concurrency problem
- Figure out how to connect ROS2 from raspberry pi with ROS2 from laptop that is connected to camera.

Demo video: https://youtube.com/shorts/2U45naUxKbg


**5. Testing, Integration, & Other:** 


## Milestone 3 Goals

