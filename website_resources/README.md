# Project Overview

Welcome to the Tetris King-Fish website! Tetris King-Fish consists of Ashley Yang, Bill Le, Oscar Bao, Sam Wisnoski, Satchel Schiavo, and the fish himself, Suketoudara. 

<p align="center">
  <img src="assets/images/suk_sr.png" alt="Suketoudara" width="400">
</p>

Our goal is to build a robot that can autonomously play Puyo Puyo Tetris 2 on the Nintendo Switch 2 by using computer vision to read the game state, 
a custom algorithm to make decisions, and a robotic arm to control the game. Success will be measured by our robot achieving a score of at least 5,000
in a single round of Tetris. As a stretch goal, we will make this arm be able to read the screen in a versus battle, allowing individuals to play tetris 
against the tetris king. 

To help us make even progress and better distribute work, we've broken this project into five components: 

**1. Computer Vision:**
Use computer vision to read the game screen by detecting the active piece, the existing grid layout, and UI elements such as next and stored pieces. This extracted game-state data will be fed into the reinforcement learning algorithm to determine the next move.

**2. Tetris Emulator:**
A custom Tetris emulator will be built to provide a controlled environment for training the reinforcement learning model. It will simulate game states, generate pieces, and apply actions so the model can learn efficiently without relying on the physical setup.

**3. Reinforcement Learning:**
Reinforcement learning will determine the robot’s optimal move by training a model using a custom Tetris simulator, reward function, and PyTorch. Although not the simplest approach, it allows the team to explore advanced decision-making methods.

**4. Robotic Arm Controls:** 
The robotic arm will follow predefined poses to press buttons on a stationary controller, using inverse kinematics and path planning to move efficiently between buttons while avoiding obstacles and minimizing movement time.

**5. Testing, Integration & Other:**
This component everything from the actual integration to testing our setup, to building the website. I guess it's kind of self explanitory based on the name of the component, but it's essentially a catch-all for things that don't fit into other categories. 


# Milestone 1 Progress Update

In our first milestone update, we will focus on each of the four components individually, since we have yet to start integration. After going through each update, we will update our MVP and discuss goals for milestone 2

## Component Progress 

**1. Computer Vision:**
We have defined the major subsystems needed for reading the screen such as creating a mask for game initialization, color mapping for piece determination, and training a YOLO model for shape smoothing any defects in our native image.
List of major project components: 
* Python libraries: cv2, numpy, ultralytics, YOLO
* Current Status: screen masking works on a demo image! Grid fidelity needs to be improved
* Next steps: Use color masking to tell game state (how many blocks are full), begin training YOLO model with block pieces, finish RGB filtering (skeleton code written)

**2. Tetris Emulator:**
We have begun building a basic Tetris environment in Python, starting with the board representation, piece definitions, and a simple class structure to manage pieces. There is still a decent ammount left to go to finish the emulator, but it's fun to work on so I hope to have it done by the end of break.

* Python libraries: numpy (board + piece arrays), time (simple loop)
* Current status: board and piece data structures defined; initial scaffold for Piece class and display loop created
* Next steps: implement piece spawning & controls, and then create a loop for generating and recording games for integratation with RL

**3. Reinforcement Learning:**
Through some research, we have identified some past attempts of trying to play Tetris autonomously, either through [RL](https://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf) or a [genetic algorithm](https://github.com/LeeYiyuan/tetrisai). It seems from the preliminary research that RL is possible, but difficult, with potential challenges being an indeterminate action space and its actual capabilities compared to a more simplified algorithm. At this point, we are still committed to doing RL as it ties to our YOGAs and proposal, but we are also ready to pivot should results look dim and we need a working model.  
* Python libraries: PyTorch
* Current status: finished installing (major challenge cleared), moving to design model architecture and layers
* Next steps: laying down model architecture and do preliminary training

**4. Robotic Arm Controls:** 
The MyCobot 280 arm has been integrated with ROS and we have gotten it to go to a specific pose using Inverse Kinematics. 
The list of major project components:
* Python libraries: numpy, modern-robotics (for inverse-kinematics), pymycobot (to interface with MyCobot 280 hardware)
* Current status: inverse kinematics works, arm integrated with ROS2
* Next steps: Get arm to press specific button and ensure that it is replicable. Test arm moving joystick.

**5. Testing, Integration, & Other:** 
We've created the website to take in a markdown file, although it's still very basic for now. We've also captured test footage of a game of tetris, allowing us to test our CV algorithms.
* Current status: We have a good start on modular testing and the website, but most of this week was individual project work and not integration. 
* Next steps: We hope to have full (semi-working) integration by milestone 2 so that we can spend our last week testing and refining our system. 


## MVP & Milestone 2
The current state of our project can best be explained by the image of Suketoudara Jr. below: 

<p align="center">
  <img src="assets/images/suk_jr.jpg" alt="Suketoudara junior" width="400">
</p>

Only a week into the project, our MVP remains *mostly* unchanged from our original goal. With further research into RL, it seems possible, but very challenging, to make a decent algorithm with RL. We still want to continue persuing reinforcement learning for the time being, but we are open to switching to a simpler heuristic model. Besides that, we still believe that the CV and Robotic arm are well within our capabilities.

For milestone 2, we hope to complete our MVP with a basic integrated pipeline. This means that we should be able to read the screen for state data, throw the state data into an algorithm, have the algorithm output an ideal move, and then have our arm input that move into the controller. We recognize that this is an ambitious goal, but we are not reaching for perfection, just a working solution. 
We expect to have to do exstensive testing and refinement between milestone 2 and 3, but if we have a good pipeline to start milestone 3, then we are setting ourselves up for success. 
