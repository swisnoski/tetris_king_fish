# Milestone 2 Progress Update



## Component Progress 

**1. Computer Vision:**


**2. Tetris Emulator:**


**3. Reinforcement Learning:**


**4. Robotic Arm Controls:** 

For this milestone, the objective is to optimize the arm speed to enable for realtime interfacing with a physical switch controller. Before the milestone, the average time for the arm to press a button and move back to the "home" position was 3s. Now, that time has decreased to 1.5s. We are still looking to decrease it even further if possible.

Here are the specific tasks achieved in this milestone:
- Physical setup with arm and controller, currently being held in place by wall putty.
- Integrating arm ROS node with `/action` topic which publishes list: [rotations, moves left/right]. Arm node receives data and performs required movement to press buttons on the controller
- Implement threading to increase task speed. Works 90% of the time. However, at times, the arm will duplicate one task, most likely a concurrency problem.

Goals til end of project:
- Fix concurrency problem
- Figure out how to connect ROS2 from raspberry pi with ROS2 from laptop that is connected to camera.

Demo video: https://youtube.com/shorts/2U45naUxKbg


**5. Testing, Integration, & Other:** 


## Milestone 3 Goals

