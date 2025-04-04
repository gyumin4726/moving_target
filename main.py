from env import TrackingEnv
from dqn import DQNAgent
from replay_buffer import ReplayBuffer
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = TrackingEnv(size=100)
agent = DQNAgent(device=device)
buffer = ReplayBuffer()

batch_size = 64
target_update_freq = 10
episodes = 500
print(f"사용 중 {device}")

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    for t in range(100):  # step limit per episode
        action = agent.select_action(state)
        next_state, reward, _ = env.step(action)
        buffer.push((state, action, reward, next_state))

        state = next_state
        total_reward += reward

        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            loss = agent.train(batch, agent.target_model)

    # 업데이트 & 로깅
    agent.update_epsilon()
    if ep % target_update_freq == 0:
        agent.target_model.load_state_dict(agent.model.state_dict())

    print(f"Episode {ep}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

# 학습 마지막 줄에 추가
torch.save(agent.model.state_dict(), "dqn_model.pth")
