from env import TrackingEnv
from dqn import DQNAgent
import cv2
import torch
import numpy as np

# 학습된 에이전트 불러오기
agent = DQNAgent()
agent.model.load_state_dict(torch.load("dqn_model.pth"))  # 저장된 모델 불러오기
agent.epsilon = 0.0  # 💡 랜덤 없이 Q-value 기반 행동만

env = TrackingEnv(size=100)

for ep in range(5):  # 5개의 에피소드 시각화
    state = env.reset()
    total_reward = 0

    for t in range(100):
        action = agent.select_action(state)
        next_state, reward, _ = env.step(action)
        total_reward += reward
        state = next_state

        # 시각화
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        ax, ay, tx, ty = state.astype(int)

        img[ty, tx] = [0, 0, 255]   # 타겟 = 빨간색
        img[ay, ax] = [255, 0, 0]   # 에이전트 = 파란색

        img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("DQN Agent Evaluation", img)
        if cv2.waitKey(30) == 27:
            break

    print(f"Eval Episode {ep}: Total Reward = {total_reward:.2f}")

cv2.destroyAllWindows()
