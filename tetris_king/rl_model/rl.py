import sys
from pathlib import Path
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tetris_king.tetris_sim.tetris_rl import Tetris_RL


class DuelingDQN(nn.Module):
    def __init__(self, board_shape, piece_info_size, action_size):
        super(DuelingDQN, self).__init__()

        self.board_height, self.board_width = board_shape

        # CNN for board - NO pooling, preserve spatial information
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 64 x 20 x 10
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128 x 20 x 10
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128 x 20 x 10
            nn.ReLU(),
            nn.Conv2d(
                128, 64, kernel_size=1
            ),  # 64 x 20 x 10 (1x1 conv to reduce channels)
            nn.ReLU(),
            nn.Flatten(),  # 64 * 20 * 10 = 12800
        )

        self.cnn_output_size = 64 * 20 * 10  # 12800

        # Piece encoder
        self.piece_encoder = nn.Sequential(
            nn.Linear(piece_info_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Combined size
        combined_size = self.cnn_output_size + 128

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(combined_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, board, piece_info):
        board = board.unsqueeze(1)  # Add channel dim

        board_features = self.conv(board)
        piece_features = self.piece_encoder(piece_info)

        combined = torch.cat([board_features, piece_features], dim=1)
        shared_features = self.shared(combined)

        value = self.value_stream(shared_features)
        advantage = self.advantage_stream(shared_features)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


# env = gym.make("CartPole-v1")
env = Tetris_RL()
board_shape = (20, 10)
state_size = 28
action_size = 44

# Hyperparameters
gamma = 0.99
epsilon_min = 0.01
epsilon_decay = 0.9999
learning_rate = 0.0005
batch_size = 128
memory_size = 100000


# Auto removing old memory above memory size
memory = deque(maxlen=memory_size)

device = torch.device("cuda")

# Policy and target network
policy_net = DuelingDQN(board_shape, state_size, action_size).to(device)
target_net = DuelingDQN(board_shape, state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
loss_fn = nn.SmoothL1Loss()


def process_state(state):
    """
    Convert raw state to board and piece_info tensors.
    Adjust this based on your actual state format.
    """
    # Assuming state contains:
    # - board: 20x10 matrix
    # - current_piece: piece id (0-6)
    # - next_pieces: list of upcoming piece ids

    # Example - adjust based on your actual state structure:
    board = np.array(state["board"])
    piece_info = state["piece_info"]
    return board, piece_info


# Explore or exploit
def get_action(board, piece_info, epsilon, valid_moves):
    if random.random() < epsilon:
        actions = [i for i in range(action_size) if valid_moves[i] == 1]
        # try:
        #     act = random.choice(actions)
        # except IndexError:
        #     print(f"Valid_moves: {valid_moves}")
        #     exit(0)
        return random.choice(actions)
    else:
        board_tensor = torch.FloatTensor(board).unsqueeze(0).to(device)
        piece_tensor = torch.FloatTensor(piece_info).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = policy_net(board_tensor, piece_tensor)

        # Mask invalid moves
        valid_mask = torch.BoolTensor(valid_moves).to(device)
        q_values[0, ~valid_mask] = -float("inf")

        return q_values.argmax().item()


# Sample experiences and update network
def replay():
    if len(memory) < batch_size:
        return

    minibatch = random.sample(memory, batch_size)

    boards, piece_infos, actions, rewards, next_boards, next_piece_infos, dones = zip(
        *minibatch
    )

    boards = torch.FloatTensor(np.array(boards)).to(device)
    piece_infos = torch.FloatTensor(np.array(piece_infos)).to(device)
    next_boards = torch.FloatTensor(np.array(next_boards)).to(device)
    next_piece_infos = torch.FloatTensor(np.array(next_piece_infos)).to(device)

    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    # Current Q values
    current_q = policy_net(boards, piece_infos).gather(1, actions)

    # Target Q values
    next_q = target_net(boards, piece_infos).max(1)[0].detach().unsqueeze(1)
    target_q = rewards + (gamma * next_q * (1 - dones))

    loss = loss_fn(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


episodes = 15000
target_update_freq = 500


def main():
    epsilon = 1.0
    lc = 0

    for episode in range(episodes):
        # reset_result = env.reset()
        # state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        board, piece_info, valid_mask = env.initialize()
        total_reward = 0

        for _ in range(1000):
            action = get_action(board, piece_info, epsilon, valid_mask)
            step_result = env.step(action + 1)

            next_board, next_piece_info, reward, done, _, valid_mask = step_result

            memory.append(
                (board, piece_info, action, reward, next_board, next_piece_info, done)
            )
            board = next_board
            piece_info = next_piece_info
            total_reward += reward

            replay()
            if done:
                break

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        lc += env.lines_cleared
        if episode % 100 == 0:
            print(
                f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}, LC: {lc}"
            )
            lc = 0

        env.reset()

    torch.save(policy_net.state_dict(), "model.pt")


if __name__ == "__main__":
    main()
