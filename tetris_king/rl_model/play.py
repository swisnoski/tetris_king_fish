import sys
from pathlib import Path
import numpy as np
import torch
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tetris_king.tetris_sim.tetris_rl import Tetris_RL
from .rl_vector import DuelingDQN  # Import your network class


def load_model(model_path, device):
    """Load a trained model."""
    board_shape = (20, 10)
    state_size = 28
    action_size = 44

    model = DuelingDQN(board_shape, state_size, action_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_action(model, board, piece_info, valid_mask, device):
    """Get the best action from the model."""
    board_tensor = torch.FloatTensor(board).unsqueeze(0).to(device)
    piece_tensor = torch.FloatTensor(piece_info).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = model(board_tensor, piece_tensor)

    # Mask invalid moves
    valid_tensor = torch.BoolTensor(valid_mask).to(device)
    q_values[0, ~valid_tensor] = -float("inf")

    return q_values.argmax().item()


def print_board(board):
    """Print the Tetris board to console."""
    print("\n" + "=" * 22)
    for row in board:
        line = "|"
        for cell in row:
            line += "█" if cell else " "
        line += "|"
        print(line)
    print("=" * 22)


def play_game(model, device, render=True, delay=0.1):
    """Play a single game and return stats."""
    env = Tetris_RL()
    board, piece_info, valid_mask = env.initialize()

    total_reward = 0
    steps = 0

    while True:
        if render:
            print_board(board)
            print(
                f"Step: {steps}, Lines: {env.lines_cleared}, Reward: {total_reward:.2f}"
            )
            time.sleep(delay)

        action = get_action(model, board, piece_info, valid_mask, device)
        next_board, next_piece_info, reward, done, _, valid_mask = env.step(action + 1)

        board = next_board
        piece_info = next_piece_info
        total_reward += reward
        steps += 1

        if done:
            if render:
                print_board(board)
                print(f"\n=== GAME OVER ===")
                print(f"Total Steps: {steps}")
                print(f"Lines Cleared: {env.lines_cleared}")
                print(f"Total Reward: {total_reward:.2f}")
            break

    return {
        "steps": steps,
        "lines_cleared": env.lines_cleared,
        "total_reward": total_reward,
    }


def evaluate(model, device, num_games=100):
    """Evaluate model over multiple games."""
    print(f"\nEvaluating over {num_games} games...")

    results = []
    for i in range(num_games):
        stats = play_game(model, device, render=False)
        results.append(stats)

        if (i + 1) % 10 == 0:
            avg_lines = np.mean([r["lines_cleared"] for r in results])
            avg_reward = np.mean([r["total_reward"] for r in results])
            print(
                f"Games: {i+1}/{num_games}, Avg Lines: {avg_lines:.2f}, Avg Reward: {avg_reward:.2f}"
            )

    # Final stats
    lines = [r["lines_cleared"] for r in results]
    rewards = [r["total_reward"] for r in results]
    steps = [r["steps"] for r in results]

    print("\n=== EVALUATION RESULTS ===")
    print(f"Games Played: {num_games}")
    print(
        f"Lines Cleared - Mean: {np.mean(lines):.2f}, Max: {np.max(lines)}, Min: {np.min(lines)}"
    )
    print(f"Total Reward  - Mean: {np.mean(rewards):.2f}, Max: {np.max(rewards):.2f}")
    print(f"Steps/Game    - Mean: {np.mean(steps):.2f}, Max: {np.max(steps)}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Play Tetris with trained model")
    parser.add_argument(
        "--model", type=str, default="model_final.pt", help="Path to model file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["play", "evaluate"],
        default="play",
        help="'play' to watch one game, 'evaluate' for stats",
    )
    parser.add_argument(
        "--games", type=int, default=100, help="Number of games for evaluation"
    )
    parser.add_argument(
        "--delay", type=float, default=0.1, help="Delay between steps (seconds)"
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else "cuda")
    print(f"Using device: {device}")

    model = load_model(args.model, device)
    print(f"Loaded model from: {args.model}")

    if args.mode == "play":
        play_game(model, device, render=True, delay=args.delay)
    else:
        evaluate(model, device, num_games=args.games)


if __name__ == "__main__":
    main()
