import sys
from pathlib import Path
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.trial import TrialState

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tetris_king.tetris_sim.tetris_rl import Tetris_RL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes, dropout_rate):
        super(DQN, self).__init__()
        layers = []
        prev_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def objective(trial):
    """Optuna objective function - returns score to maximize."""

    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.99, 0.9999)
    target_update_freq = trial.suggest_int("target_update_freq", 50, 1000)
    memory_size = trial.suggest_categorical("memory_size", [50000, 100000, 200000])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3)

    # Network architecture
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_sizes = []
    for i in range(n_layers):
        hidden_sizes.append(
            trial.suggest_categorical(f"hidden_size_{i}", [64, 128, 256, 512])
        )

    # Training
    env = Tetris_RL()
    state_size = 38
    action_size = 44

    memory = deque(maxlen=memory_size)

    policy_net = DQN(state_size, action_size, hidden_sizes, dropout_rate).to(device)
    target_net = DQN(state_size, action_size, hidden_sizes, dropout_rate).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    epsilon = 1.0
    epsilon_min = 0.01
    rewards_history = []
    episodes = 1000  # Shorter for tuning

    for episode in range(episodes):
        state, valid_mask = env.initialize()
        total_reward = 0

        for _ in range(1000):
            if random.random() < epsilon:
                actions = [i for i in range(action_size) if valid_mask[i] == 1]
                action = random.choice(actions)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                valid_tensor = torch.BoolTensor(valid_mask).to(device)
                q_values[0, ~valid_tensor] = -float("inf")
                action = q_values.argmax().item()

            next_state, reward, done, _, valid_mask = env.step(action + 1)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) >= batch_size:
                minibatch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                current_q = policy_net(states).gather(1, actions)
                next_q = target_net(next_states).max(1)[0].detach().unsqueeze(1)
                target_q = rewards + (gamma * next_q * (1 - dones))

                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        rewards_history.append(total_reward)
        env.reset()

        # Pruning - stop bad trials early
        if episode % 100 == 0 and episode > 0:
            intermediate_value = np.mean(rewards_history[-100:])
            trial.report(intermediate_value, episode)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return np.mean(rewards_history[-100:])


def main():
    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name="tetris_dqn_tuning",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=300),
    )

    # Run optimization
    study.optimize(objective, n_trials=50, timeout=3600 * 4)  # 4 hours max

    # Results
    print("\n=== BEST TRIAL ===")
    trial = study.best_trial
    print(f"Score: {trial.value}")
    print("Parameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # Save best params
    import json

    with open("best_params.json", "w") as f:
        json.dump(trial.params, f, indent=2)

    # Visualization (optional)
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html("param_importance.html")

        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html("optimization_history.html")
    except:
        pass


if __name__ == "__main__":
    main()
