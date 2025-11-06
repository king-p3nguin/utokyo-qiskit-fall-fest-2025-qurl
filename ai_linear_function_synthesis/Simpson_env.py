import numpy as np
from .env import AILinearFunctionSynthesis 

class AILinearFunctionSynthesisSimpsonReward(AILinearFunctionSynthesis):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.previous_distance = 0.0

    @staticmethod
    def _compute_simpson_distance(state_a: np.ndarray, state_b: np.ndarray) -> float:

        state_a_bool = state_a.astype(np.bool_)
        state_b_bool = state_b.astype(np.bool_)

        intersection = (state_a_bool & state_b_bool).sum()
        
        size_a = state_a_bool.sum()
        size_b = state_b_bool.sum()
        min_size = min(size_a, size_b)
        
        if min_size == 0:
            if size_a == 0 and size_b == 0:
             return 0.0
            else:
                return 1.0 
        
        similarity = float(intersection) / min_size
        
        return 1.0 - similarity

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        
        desired_goal = np.eye(self.num_qubits, dtype=np.bool_)

        self.previous_distance = self._compute_simpson_distance(self.state, desired_goal)
        
        return obs, info

    def _compute_reward(self, achieved_goal: np.ndarray) -> np.float32:
        
        desired_goal = np.eye(self.num_qubits, dtype=np.bool_)

        current_distance = self._compute_simpson_distance(achieved_goal, desired_goal)

        reward = self.previous_distance - current_distance
        
        self.previous_distance = current_distance

        reward -= self.num_cnots * 0.01
        
        return np.float32(reward)