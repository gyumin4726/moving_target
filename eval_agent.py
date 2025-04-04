from env import TrackingEnv
from dqn import DQNAgent
import cv2
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = DQNAgent(device=device)
agent.model.load_state_dict(torch.load("dqn_model.pth", map_location=device))  # ✅ GPU에서 로드
agent.model.to(device)  
agent.epsilon = 0.0 # 탐험 없음


env = TrackingEnv(size=100)

agent_history = []
target_history = []

print(f"사용 중 {device}")

for ep in range(5):  # 5개 에피소드 실행
    state = env.reset()
    total_reward = 0
    agent_history.clear()
    target_history.clear()

    for t in range(100):
        action = agent.select_action(state)
        next_state, reward, _ = env.step(action)
        total_reward += reward
        state = next_state

        ax, ay, tx, ty = state.astype(int)

        agent_history.append((ax, ay))
        target_history.append((tx, ty))

        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        for i in range(1, len(agent_history)):
            cv2.line(img, agent_history[i - 1], agent_history[i], color=(255, 0, 0), thickness=1)
        for i in range(1, len(target_history)):
            cv2.line(img, target_history[i - 1], target_history[i], color=(0, 0, 255), thickness=1)

        img[ty, tx] = [0, 0, 150]    # 타겟
        img[ay, ax] = [150, 0, 0]    # 에이전트

        img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("DQN Agent Evaluation (Path)", img)
        if cv2.waitKey(30) == 27:
            break

    print(f"Eval Episode {ep}: Total Reward = {total_reward:.2f}")

cv2.destroyAllWindows()
