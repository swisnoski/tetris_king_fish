import gymnasium as gym
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from tetris_king.tetris_sim.tetris_rl import Tetris_RL


# Defining the model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 69)
        self.fc2 = nn.Linear(69, 69)
        self.fc3 = nn.Linear(69, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# env = gym.make("CartPole-v1")
env = Tetris_RL()
state_size = 14
action_size = 44

# Hyperparameters
gamma = 0.99
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 64
memory_size = 10000

# Auto removing old memory above memory size
memory = deque(maxlen=memory_size)

device = torch.device("cuda")

# Policy and target network
policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()


# Explore or exploit
def get_action(state, epsilon, valid_moves):
    if random.random() < epsilon:
        return random.choice(range(action_size))
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state)
        # Set Q-values of illegal moves to Negative Infinity
        q_values[~valid_moves] = -float("inf")

        # Now select the action
        # action = torch.argmax(q_values)
        return q_values.argmax().item()


# Sample experiences and update network
def replay():
    if len(memory) < batch_size:
        return

    minibatch = random.sample(memory, batch_size)

    states, actions, rewards, next_states, dones = zip(*minibatch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    # Current Q values
    current_q = policy_net(states).gather(1, actions)

    # Target Q values
    next_q = target_net(next_states).max(1)[0].detach().unsqueeze(1)
    target_q = rewards + (gamma * next_q * (1 - dones))

    loss = loss_fn(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


episodes = 500
target_update_freq = 10


def main():
    epsilon = 1.0

    for episode in range(episodes):
        # reset_result = env.reset()
        # state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        state, valid_mask = env.initialize()
        total_reward = 0

        for _ in range(500):
            action = get_action(state, epsilon, valid_mask)
            step_result = env.step(action)

            # if len(step_result) == 5:
            #     next_state, reward, terminated, truncated, _ = step_result
            #     done = terminated or truncated
            # else:
            #     next_state, reward, done, _ = step_result
            next_state, reward, done, _, valid_mask = step_result

            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            replay()
            if done:
                break

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        env.close()

        print(
            f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}"
        )

    torch.save(policy_net.state_dict(), "model.pt")


if __name__ == "__main__":
    main()
