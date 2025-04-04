import numpy as np

class TrackingEnv:
    def __init__(self, size=100):
        self.size = size
        self.reset()
 
    def reset(self):
        # 타겟과 에이전트 위치 초기화
        self.agent_pos = np.array([np.random.randint(self.size), np.random.randint(self.size)])
        self.target_pos = np.array([np.random.randint(self.size), np.random.randint(self.size)])
        return self._get_state()

    def _get_state(self):
        # 좌표 기반 상태 반환
        return np.concatenate([self.agent_pos, self.target_pos])

    def step(self, action):
        # action: 0=up, 1=down, 2=left, 3=right
        moves = {
            0: np.array([0, -1]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([1, 0]),
        }
        move = moves.get(action, np.array([0, 0]))
        self.agent_pos = np.clip(self.agent_pos + move, 0, self.size - 1)

        # 타겟 랜덤 이동 (8방향 중 하나 or 정지)
        target_move = np.random.randint(-1, 2, size=2)
        self.target_pos = np.clip(self.target_pos + target_move, 0, self.size - 1)

        # 거리 기반 보상
        distance = np.linalg.norm(self.agent_pos - self.target_pos)
        reward = -distance / self.size

        done = False  # 일단은 done 없이 무한 시뮬레이션
        return self._get_state(), reward, done
