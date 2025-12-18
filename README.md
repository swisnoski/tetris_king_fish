# Tetris King
Sam, Bill, Ashley, Satchel, Oscar

## Project Goal

Tetris King is our final robotics project for Computation Robotics (Fall '25). Tetris King plays real, physical Tetris end-to-end. We use a camera to watch the game, a simulator to understand what’s happening on the board, (theoretically) an AI model to decide the best move, and a robotic arm to press the buttons on a joycon.

This repository contains the code. If you are interested in learning more, please visit: https://tetrisking.oscarbao.com/

You can also view a live demo here: https://www.youtube.com/watch?v=sDlx_0leq-A 

## High-level Overview

The system works as a pipeline:

1. **Computer Vision** watches a live camera feed of a Tetris screen and figures out the game state.
2. A **Tetris emulator** mirrors the game internally so moves can be tested safely.
3. An **AI agent** (heuristic-based and reinforcement learning) chooses what move to make.
4. A **robotic arm** physically presses buttons on a controller to execute that move.
