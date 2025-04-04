import numpy as np
import random

class TrackingEnv:
    def __init__(self, size=100):
        self.size = size
        self.reset()

    def reset(self):
        # 타겟과 에이전트 위치 초기화
        self.agent_pos = np.array([np.random.randint(self.size), np.random.randint(self.size)])
        self.target_pos = np.array([np.random.randint(self.size), np.random.randint(self.size)])

        # 타겟의 목표 지점 설정
        self.target_goal = np.random.randint(0, self.size, size=2)

        return self._get_state()

    def _get_state(self):
        # 좌표 기반 상태 반환
        return np.concatenate([self.agent_pos, self.target_pos])

    def step(self, action):
        # 에이전트 이동
        moves = {
            0: np.array([0, -1]),  # 상
            1: np.array([0, 1]),   # 하
            2: np.array([-1, 0]),  # 좌
            3: np.array([1, 0]),   # 우
        }
        move = moves.get(action, np.array([0, 0]))
        self.agent_pos = np.clip(self.agent_pos + move, 0, self.size - 1)

        # 목표 지점에 도달하면 새로운 목표 설정
        if np.linalg.norm(self.target_goal - self.target_pos) < 1:
            self.target_goal = np.random.randint(0, self.size, size=2)

        # 목표 방향 + 랜덤 흔들림 이동
        direction = self.target_goal - self.target_pos
        if np.linalg.norm(direction) != 0:
            move = np.round(direction / np.linalg.norm(direction)).astype(int)

            # 🔥 랜덤 noise 추가 (10% 확률로 x/y방향 살짝 흔들림)
            noise = np.random.choice([-1, 0, 1], size=2, p=[0.1, 0.8, 0.1])
            move = move + noise

            self.target_pos = np.clip(self.target_pos + move, 0, self.size - 1)

        # 거리 기반 보상
        distance = np.linalg.norm(self.agent_pos - self.target_pos)
        reward = -distance / self.size

        done = False
        return self._get_state(), reward, done

