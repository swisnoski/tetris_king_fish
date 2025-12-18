## Learning goals 

#### Ashley:

At the start of this project, my learning goals were to use C++ for one of the features of this project, such as computer vision module, or at least have a Python version that goes on the final integrated version and a separate C++ version that I developed in isolation. For the second goal, I strived to have a clear conceptual understanding of each different component of this project, as they each touched a different topic of robotics that I wanted to have a strong foundational understanding of.

Looking back, I realize that my expectations of what I would accomplish for my learning goals ended up differing quite a bit from what I ended up strengthing skills in and developing; rather than learning a new language and gaining broad topics understanding, this project was more of a deep dive into computer vision and how to implement software system design principles and practices for a real-time live system. While unexpected, I found it really rewarding to extend my software skills and get to run into actual situations that exposed bugs or detection consistency issues, iterate on my code to address them, build testing functions and examples again, and finally create a satisfactorily reliable system by the end of the project. 

#### Bill: 
My two technical YOGA goals this semester are mastering robotic software architecture and engaging with open-source libraries. This project provided avenues to accomplish my first goal. There are a variety of nodes and tasks that need to be organized into a seamless and intuitive structure. Due to the real time nature of the problem, there is a need for optimizing performance. I think architecture will play an important role in that. As for my second goal, I haven’t made any progress this semester. However, I am still immensely interested in learning how to contribute to an open-source library. The two most likely open-source library options are reinforcement learning and robotics motion planning. Throughout the project, I will continuously reflect on how I could contribute to these libraries.

#### Sam: 
My YOGAs are largely related to increasing my knowledge of robotic systems, specifically, “robots that use cameras to take in data, and can physically respond to the world around them”. Conveniently, this project matches that description exactly — however, the next step is making sure that I am actually engaging with the CV and physical manipulation areas of the project.   

Looking back, I think I achieved exactly what I was looking for out of this project. I ended up working mainly on the tetris simulator, but also got to help occasionally with RL and CV. I ended up finishing the simulator ahead of schedule and was able to help debug in other areas, which is definitely a skill I've needed some practice with. 

## Who did what 

#### Ashley:
I worked on the computer vision game state detection, which involved initializing the grid cells center coordinates to check for fill, outputting the game state matrix of pieces filled through HSV color thresholding, scrubbing the current piece from the fill grid game state, and detecting and outputting the current piece that the player has. In addition, I put together the overall computer vision architecture pipeline to chain together the computer vision sequence steps and handle user action at the start of the program, the video capture feed, and output to the rest of the overall program pipeline.

#### Bill: 
I worked on the arm, made the arm hit buttons. This involved calculating the forward and reverse kinematics, implementing control via a socket, and working heavily with trial and error. 

#### Sam: 
I built the tetris simulators (tetris, tetris_max, tetris_rl) in order to practice my systems coding skills and because I thought it would be a fun challenge. I also helped debug and assisted with integration, as we all did. 

## Challenges 

**Computer Vision**
While there was the expectation going in that the real-time gameplay and screen glare aspects were going to be challenges for computer vision, the extent of the reliability hurdles, error handling, and code scoping were still tough challenges during the development of the computer vision. Since the rest of the overal pipeline operated under the assumption of a very reliable computer vision output, seemingly small changes like another light turned on in the room would affect the color and piece detection enough to have a large effect on the system. For example, at one point, the current piece detection would output a wrong piece once out of the multiple other correct hits it got on the same piece; while in isolation, this wouldn't seem like a large issue, this affected the model and physical arm performance enough to not be functional. This required a multitude of tuning and testing rounds to adjust our HSV piece detecting and mapping, as well as implementing a way to handle the most frequent match from the detection.

Overall, turning our original plan of the computer vision pipeline, which in the end remained sound, into the actual specifics of how to handle a real-time gameplay situation was a challenge to tackle. A prime example for this would be our original plan to initialize the grid just once at the beginning of the overall program loop, since the coordinates would be constant from our static camera. However, in development, amidst quick testing scenarios with cameras and images, we found it infeasible for the initialization to run just once and be able to snap accurately into place. So, we ran the grid detection every loop so we could develop and see the rest of the CV pipeline; in actual gameplay testing, this created inconsistent game state detection. Yet, doing the grid detection just once would run into the same previous error, especially when we found that the initial frame grabbed from our webcam feed was a black frame, and that in real usage, the webcam is not always perfectly set in place by the user. In the end, after testing and observing the behavior, our implementation was to let the user try the grid detection multiple times on key press, until they were satisifed with the grid points they saw drawn on the frame, before jumping into the program loop — this was overall our same original idea, but only after multiple rounds of developing and testing did we implement a way that translated into a sensible and usable component.

**Tetris Simulation** 
Because this project was so spread out, some design decisions I made at the beginning of the project blindsided me as we approached the end. Although the simulator didn't have too many problems contained in itself, once we began integration we ran into trouble. At the very end of the project, at 3AM, I spent half an hour trying to debug what I thought was a programtically perfect system, before remembering that I should have been marking cells as '2' instead of as '1'. 

**Robotic Arm**
One problem we faced were the compute power of the Raspi 3 Model B. We first discovered this problem when running the numerical IK on it. When ran on the laptop, the numerical IK took 0.02 to 0.05 seconds. However on the Raspi, it took 0.3 seconds. It was so much slower, we had to calculate the IK before hand and input manual angle solutions to the Raspi. Therefore, a setup where the arm, a camera, and the whole pipeline running the Raspi 3 Model B was out of the question.

## Limitations 

**Computer Vision**

Currently, the big limitations with the computer vision is how the camera placement needs to be somewhat specific, and how drastically the lighting in the room affects the performance of the game state detection. Even using HSV for the color matching, which is more resistant than RBG to differences in lighting, and tuning the matching calculation weights to have a 8:1 hue to value ratio (saturation cut out completely), the inherent functionality of a camera looking at an electronic game screen means that the HSV values still differ significantly under seemingly small lighting changes like an extra lamp being turned on. This means that the computer vision is constrained to working reliably only under the specific environment we set up the system in, with the same screen and lighting, unless the HSV values are re-tuned.

**Arm** 
The arm can only move so quickly, so as the tetris pieces get higher, it is unable to move pieces over in time. Additonally, without a more solid base, the arm can occasionally miss it's target. This is especially bad, since a single missed tetris piece will often destroy an otherwise perfectly good run. 

## Potential improvements

**Computer Vision**

For this project, we constrained scope to get camera input that's centered largely on the gameplay grid we were working with. A potential improvement to this could be to build in screen detection to crop off the noise around the TV screen, which would allow for a less sensitive positioning of the camera.

Additionally, as mentioned, lighting conditions were a significant limitation for a robust computer vision component; given more time, a potential improvment to address this issue would be to experiment with other techniques to detect fill or current piece, in addition to or replacing the single HSV reference point for each tetromino. Possibilities include reversing the fill detection to look for a match to the dark empty cells instead, and adding in shape matching for current piece detection to incorporate extra validation.

## AI use disclosure 
#### Sam: I used ChatGPT throughout the project as I debugged code, usually only when I would get a terminal error. I did not use ChatPGT to write any code. I do not have the exact questions I asked, but I am sure I can find them if needed. 
