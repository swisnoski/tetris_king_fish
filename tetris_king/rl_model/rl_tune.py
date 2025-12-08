import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv
import optuna
from collections import deque
from tetris_king.tetris_sim.tetris_rl import Tetris_RL

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ENVS = 32  # Keep 32 to utilize CPU cores
DB_URL = "sqlite:///tetris_sweep.db"  # Shared database file
STUDY_NAME = "tetris_l40s_optimization"


# --- 1. THE WRAPPER (Now with Normalization!) ---
class TetrisGymWrapper(gym.Env):
    def __init__(self):
        self.game = Tetris_RL()
        self.action_space = spaces.Discrete(44)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=1, shape=(20, 10), dtype=np.float32),
                "piece_info": spaces.Box(low=0, high=1, shape=(28,), dtype=np.float32),
                "valid_mask": spaces.Box(low=0, high=1, shape=(44,), dtype=bool),
            }
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        board, piece_info, valid_mask = self.game.initialize()
        return self._format_obs(board, piece_info, valid_mask), {}

    def step(self, action):
        next_board, next_piece, reward, done, _, valid_mask = self.game.step(
            int(action) + 1
        )
        # CRITICAL: Reward Shaping is usually done here if not in the engine
        # But we assume your engine handles the heuristic calculation
        return (
            self._format_obs(next_board, next_piece, valid_mask),
            float(reward),
            bool(done),
            False,
            {},
        )

    def _format_obs(self, board, piece, mask):
        # NORMALIZATION: Divide by 2.0 to get 0.0 to 1.0 range
        return {
            "board": (np.array(board, dtype=np.float32) / 2.0),
            "piece_info": np.array(piece, dtype=np.float32),
            "valid_mask": np.array(mask, dtype=bool),
        }


# --- 2. THE NETWORK (Hybrid Architecture) ---
class DuelingDQN(nn.Module):
    def __init__(self, action_size):
        super(DuelingDQN, self).__init__()
        # CNN (1x1 Conv trick included)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.cnn_out = 32 * 20 * 10  # 6400

        self.piece_enc = nn.Sequential(nn.Linear(28, 32), nn.ReLU(inplace=True))

        # Combined
        self.shared = nn.Sequential(nn.Linear(6400 + 32, 512), nn.ReLU(inplace=True))
        self.val = nn.Linear(512, 1)
        self.adv = nn.Linear(512, action_size)

    def forward(self, board, piece):
        # board is already normalized [0,1] from wrapper
        x = self.conv(board.unsqueeze(1))
        p = self.piece_enc(piece)
        combined = self.shared(torch.cat([x, p], dim=1))
        v = self.val(combined)
        a = self.adv(combined)
        return v + (a - a.mean(dim=1, keepdim=True))


# --- 3. GPU REPLAY BUFFER ---
class FastReplayBuffer:
    def __init__(self, max_size, num_envs):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        # Pre-allocate on GPU
        self.boards = torch.zeros(
            (max_size, 20, 10), dtype=torch.float32, device=DEVICE
        )
        self.pieces = torch.zeros((max_size, 28), dtype=torch.float32, device=DEVICE)
        self.actions = torch.zeros((max_size, 1), dtype=torch.long, device=DEVICE)
        self.rewards = torch.zeros((max_size, 1), dtype=torch.float32, device=DEVICE)
        self.next_boards = torch.zeros(
            (max_size, 20, 10), dtype=torch.float32, device=DEVICE
        )
        self.next_pieces = torch.zeros(
            (max_size, 28), dtype=torch.float32, device=DEVICE
        )
        self.dones = torch.zeros((max_size, 1), dtype=torch.float32, device=DEVICE)

    def add(self, obs, act, rew, next_obs, done):
        # Vectorized add
        n = obs["board"].shape[0]  # usually 32
        idx = self.ptr
        end = idx + n
        if end <= self.max_size:
            self.boards[idx:end] = torch.as_tensor(obs["board"], device=DEVICE)
            self.pieces[idx:end] = torch.as_tensor(obs["piece_info"], device=DEVICE)
            self.actions[idx:end] = torch.as_tensor(act, device=DEVICE).unsqueeze(1)
            self.rewards[idx:end] = torch.as_tensor(rew, device=DEVICE).unsqueeze(1)
            self.next_boards[idx:end] = torch.as_tensor(
                next_obs["board"], device=DEVICE
            )
            self.next_pieces[idx:end] = torch.as_tensor(
                next_obs["piece_info"], device=DEVICE
            )
            self.dones[idx:end] = torch.as_tensor(done, device=DEVICE).unsqueeze(1)
            self.ptr = (self.ptr + n) % self.max_size
            self.size = min(self.size + n, self.max_size)
        else:
            # Simple overflow handling: just drop extra for speed in this sweep
            pass

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=DEVICE)
        return (
            self.boards[idx],
            self.pieces[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_boards[idx],
            self.next_pieces[idx],
            self.dones[idx],
        )


# --- 4. OPTUNA OBJECTIVE ---
def objective(trial):
    # --- HYPERPARAMETERS TO TUNE ---
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2048, 4096, 8192])
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    train_freq = trial.suggest_int(
        "train_freq", 1, 10
    )  # Train X times per collection burst

    # Setup
    policy = DuelingDQN(44).to(DEVICE)
    target = DuelingDQN(44).to(DEVICE)
    target.load_state_dict(policy.state_dict())
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Memory (L40S has RAM to spare, let's go big)
    memory = FastReplayBuffer(500_000, NUM_ENVS)

    # Env
    env_fns = [lambda: TetrisGymWrapper() for _ in range(NUM_ENVS)]
    vec_env = AsyncVectorEnv(env_fns)

    # Training Loop
    obs, _ = vec_env.reset()
    total_steps = 0
    MAX_STEPS = 2_000_000  # 2M steps is a good "Performance Test"

    # Epsilon Schedule (Linear)
    eps_start, eps_end = 1.0, 0.05
    explore_steps = MAX_STEPS * 0.5

    running_reward = deque(maxlen=100)

    try:
        while total_steps < MAX_STEPS:
            # 1. Calculate Epsilon
            epsilon = max(eps_end, eps_start - (total_steps / explore_steps))

            # 2. Collect Data (Burst of 20 steps x 32 envs = 640 frames)
            for _ in range(20):
                # Vectorized Action Selection
                boards_t = torch.as_tensor(obs["board"], device=DEVICE)
                pieces_t = torch.as_tensor(obs["piece_info"], device=DEVICE)
                masks_t = torch.as_tensor(obs["valid_mask"], device=DEVICE)

                # Epsilon Logic
                if random.random() < epsilon:
                    # Random valid moves (CPU side for speed)
                    actions = []
                    for mask in obs["valid_mask"]:
                        valid = np.where(mask)[0]
                        actions.append(np.random.choice(valid) if len(valid) > 0 else 0)
                    actions = np.array(actions)
                else:
                    with torch.no_grad():
                        q = policy(boards_t, pieces_t)
                        q[~masks_t] = -float("inf")
                        actions = q.argmax(dim=1).cpu().numpy()

                next_obs, rews, dones, _, _ = vec_env.step(actions)

                # Store
                memory.add(obs, actions, rews, next_obs, dones)
                obs = next_obs
                total_steps += NUM_ENVS

                # Track Reward (average across batch)
                if np.any(dones):
                    # In vector env, rewards are per-step.
                    # We approximate "performance" by tracking raw rewards received.
                    running_reward.append(np.mean(rews))

            # 3. Train (Burst)
            if memory.size > batch_size:
                for _ in range(train_freq):  # Use the tuned frequency
                    b_brd, b_pcs, b_act, b_rew, b_nxt_brd, b_nxt_pcs, b_don = (
                        memory.sample(batch_size)
                    )

                    cur_q = policy(b_brd, b_pcs).gather(1, b_act)
                    with torch.no_grad():
                        nxt_act = policy(b_nxt_brd, b_nxt_pcs).argmax(
                            dim=1, keepdim=True
                        )
                        nxt_q = target(b_nxt_brd, b_nxt_pcs).gather(1, nxt_act)
                        # Bellman with 'done' handling
                        tgt_q = b_rew + (gamma * nxt_q * (1 - b_don))

                    loss = nn.SmoothL1Loss()(cur_q, tgt_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # 4. Target Update
            if total_steps % 2000 < NUM_ENVS:
                target.load_state_dict(policy.state_dict())

            # 5. Report to Optuna (Pruning)
            if total_steps % 50000 < NUM_ENVS:
                avg_rew = np.mean(running_reward) if len(running_reward) > 0 else 0
                trial.report(avg_rew, total_steps)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    finally:
        vec_env.close()

    return np.mean(running_reward)


if __name__ == "__main__":
    # Initialize DB if needed
    storage = optuna.storages.RDBStorage(url=DB_URL)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=200000),
    )

    print(f"Starting worker on {DEVICE}...")
    # Run 10 trials per job
    study.optimize(objective, n_trials=10)
