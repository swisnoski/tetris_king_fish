import sys
from pathlib import Path
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

# --- IMPORT YOUR CUSTOM ENV ---
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tetris_king.tetris_sim.tetris_rl import Tetris_RL


# --- 1. THE GYM WRAPPER ---
class TetrisGymWrapper(gym.Env):
    def __init__(self):
        self.game = Tetris_RL()
        self.action_space = gym.spaces.Discrete(44)
        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.Box(
                    low=0, high=1, shape=(20, 10), dtype=np.float32
                ),
                "piece_info": gym.spaces.Box(
                    low=0, high=1, shape=(28,), dtype=np.float32
                ),
                "valid_mask": gym.spaces.Box(low=0, high=1, shape=(44,), dtype=bool),
            }
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        board, piece_info, valid_mask = self.game.initialize()
        return {
            "board": np.array(board, dtype=np.float32),
            "piece_info": np.array(piece_info, dtype=np.float32),
            "valid_mask": np.array(valid_mask, dtype=bool),
        }, {}

    def step(self, action):
        # Convert scalar array to int if needed, add 1 if your env expects 1-based index
        next_board, next_piece, reward, done, _, valid_mask = self.game.step(
            int(action) + 1
        )

        obs = {
            "board": np.array(next_board, dtype=np.float32),
            "piece_info": np.array(next_piece, dtype=np.float32),
            "valid_mask": np.array(valid_mask, dtype=bool),
        }
        return obs, float(reward), bool(done), False, {}


# --- 2. YOUR NETWORK (Unchanged) ---
class DuelingDQN(nn.Module):
    def __init__(self, board_shape, piece_info_size, action_size):
        super(DuelingDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.cnn_output_size = 6400
        self.piece_encoder = nn.Sequential(
            nn.Linear(piece_info_size, 32), nn.ReLU(inplace=True)
        )
        self.shared = nn.Sequential(
            nn.Linear(self.cnn_output_size + 32, 512),
            nn.ReLU(inplace=True),
        )
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, action_size)

    def forward(self, board, piece_info):
        board = board.unsqueeze(1)  # (Batch, 1, 20, 10)
        board_features = self.conv(board)
        piece_features = self.piece_encoder(piece_info)
        combined = torch.cat([board_features, piece_features], dim=1)
        shared_features = self.shared(combined)
        val = self.value_stream(shared_features)
        adv = self.advantage_stream(shared_features)
        return val + (adv - adv.mean(dim=1, keepdim=True))


# --- HYPERPARAMETERS ---
NUM_ENVS = 32  # Parallel environments (adjust based on CPU cores)
board_shape = (20, 10)
state_size = 28
action_size = 44
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9999  # Decay slower because we take more steps per second
learning_rate = 0.0003  # Lower LR for larger batches
batch_size = 1024  # Increased batch size for L40S
memory_size = 200000

device = torch.device("cuda")

# --- INITIALIZATION ---
policy_net = DuelingDQN(board_shape, state_size, action_size).to(device)
target_net = DuelingDQN(board_shape, state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
loss_fn = nn.SmoothL1Loss()
memory = deque(maxlen=memory_size)

# --- HELPER FUNCTIONS ---


def get_actions_batch(obs, epsilon):
    """
    Select actions for the entire batch of environments at once.
    obs is a dict of batched arrays (NUM_ENVS, ...)
    """
    boards = torch.FloatTensor(obs["board"]).to(device)
    piece_infos = torch.FloatTensor(obs["piece_info"]).to(device)
    valid_masks = torch.BoolTensor(obs["valid_mask"]).to(device)

    actions = np.zeros(NUM_ENVS, dtype=int)

    # 1. Epsilon Greedy Logic (Vectorized)
    # Generate random numbers for the whole batch
    rand_vals = np.random.random(NUM_ENVS)
    explore_indices = np.where(rand_vals < epsilon)[0]
    exploit_indices = np.where(rand_vals >= epsilon)[0]

    # Exploration (Random Valid Move)
    if len(explore_indices) > 0:
        masks_cpu = obs["valid_mask"][explore_indices]
        for idx, mask in zip(explore_indices, masks_cpu):
            valid_indices = np.where(mask)[0]
            if len(valid_indices) > 0:
                actions[idx] = np.random.choice(valid_indices)
            else:
                actions[idx] = 0  # Fallback if no valid moves (shouldn't happen)

    # Exploitation (Network Prediction)
    if len(exploit_indices) > 0:
        with torch.no_grad():
            # Run the network on the WHOLE batch (it's faster to run 32 than 10)
            # We filter later, or just run all and pick specific indices
            q_values = policy_net(boards, piece_infos)

            # Mask invalid moves
            q_values[~valid_masks] = -float("inf")

            # Get argmax
            best_actions = q_values.argmax(dim=1).cpu().numpy()

            # Assign to the exploit indices
            actions[exploit_indices] = best_actions[exploit_indices]

    return actions


def replay():
    if len(memory) < batch_size:
        return

    minibatch = random.sample(memory, batch_size)

    # Unpack minibatch
    boards, piece_infos, actions, rewards, next_boards, next_piece_infos, dones = zip(
        *minibatch
    )

    # Convert to Tensor
    boards = torch.FloatTensor(np.array(boards)).to(device)
    piece_infos = torch.FloatTensor(np.array(piece_infos)).to(device)
    next_boards = torch.FloatTensor(np.array(next_boards)).to(device)
    next_piece_infos = torch.FloatTensor(np.array(next_piece_infos)).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    # Double DQN Logic
    # 1. Selection (Policy Net)
    current_q = policy_net(boards, piece_infos).gather(1, actions)

    with torch.no_grad():
        # Get best actions for next state using Policy Net
        next_q_policy = policy_net(next_boards, next_piece_infos)
        # We don't have valid masks for next states in memory easily,
        # but generally the policy net learns to avoid invalid moves naturally.
        # Ideally, store next_masks in memory too.
        next_actions = next_q_policy.argmax(dim=1, keepdim=True)

        # 2. Evaluation (Target Net)
        next_q_target = target_net(next_boards, next_piece_infos).gather(
            1, next_actions
        )
        target_q = rewards + (gamma * next_q_target * (1 - dones))

    loss = loss_fn(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# --- MAIN LOOP ---
def main():
    global epsilon

    # 1. Create Vectorized Environment
    # This spawns processes!
    env_fns = [lambda: TetrisGymWrapper() for _ in range(NUM_ENVS)]
    vec_env = AsyncVectorEnv(env_fns)

    # 2. Initial Reset
    # "obs" is now a dict of arrays: obs['board'] shape is (32, 20, 10)
    obs, _ = vec_env.reset()

    total_steps = 0
    target_update_freq = 2000

    print(f"Training started on {device} with {NUM_ENVS} environments...")

    try:
        while total_steps < 2000000:  # Train for 2M steps (or use infinite loop)

            # 1. Get Batch of Actions
            actions = get_actions_batch(obs, epsilon)

            # 2. Step the Environment (All 32 at once)
            # vec_env auto-resets sub-envs that are done!
            # next_obs is the NEW state (reset if done)
            # infos contains the "terminal state" info if done
            next_obs, rewards, dones, truncs, infos = vec_env.step(actions)

            # 3. Store in Memory
            # We loop through the batch to store them individually
            # (There are faster ways, but this is fine for now)
            for i in range(NUM_ENVS):
                # Important: If done, next_obs[i] is the NEW reset state.
                # The "real" final state is in infos['final_observation'][i] if we needed it.
                # For standard DQN, using the reset state as "next_state" with done=True is handled by (1-done)

                # However, for correct transitions, if done[i] is True, next_obs[i] is irrelevant for Q-value.
                # We save it anyway because 'done' flag handles the math.

                memory.append(
                    (
                        obs["board"][i],
                        obs["piece_info"][i],
                        actions[i],
                        rewards[i],
                        next_obs["board"][i],
                        next_obs["piece_info"][i],
                        dones[i],
                    )
                )

            # 4. Train
            if len(memory) > batch_size * 2:  # Wait for some data
                replay()

                # Optional: Replay multiple times per step if GPU is bored
                # replay()

            # 5. Updates
            obs = next_obs
            total_steps += NUM_ENVS  # We took 32 steps in total

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            if total_steps % target_update_freq < NUM_ENVS:
                target_net.load_state_dict(policy_net.state_dict())
                print(f"Steps: {total_steps}, Epsilon: {epsilon:.4f}")
                torch.save(policy_net.state_dict(), "model_checkpoint.pt")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        vec_env.close()
        torch.save(policy_net.state_dict(), "model_final.pt")


if __name__ == "__main__":
    main()
