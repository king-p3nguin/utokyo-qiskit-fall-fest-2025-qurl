import numpy as np
from gymnasium import Env, spaces
from qiskit.transpiler import CouplingMap

import wandb

from .circuit import *


class AILinearFunctionSynthesis(Env):
    """ref: stable_baselines3.common.envs.bit_flipping_env.BitFlippingEnv"""

    def __init__(
        self,
        coupling_map: CouplingMap,
        eval_batch_size: int = 100,
        success_rate_threshold: float = 0.8,
        wandb_log: bool = False,
        render_mode: str = "human",
    ):
        super().__init__()
        self.render_mode = render_mode
        self.wandb_log = wandb_log

        # track cnot counts while training
        self.num_cnots = 0

        # track applied cnot gates
        self.cnot_gates = []

        # qubit coupling map
        self.coupling_map = coupling_map
        # number of qubits
        self.num_qubits = coupling_map.size()
        for control, target in self.coupling_map:
            if (
                not (0 <= control < self.num_qubits or 0 <=
                     target < self.num_qubits)
            ) or (control == target):
                raise ValueError(
                    f"Invalid qubit index {[control, target]}.0 "
                    "The qubit index must be in the range [0, num_qubits) "
                    "and control != target."
                )
        # matrix shape as an image (gray scale = single channel)
        self.image_shape = (1, self.num_qubits, self.num_qubits)
        # number of episodes before increasing the difficulty
        self.eval_batch_size = eval_batch_size
        # number of episodes
        self.total_count = 0
        # number of successful episodes
        self.success_count = 0
        # threshold for the success rate
        self.success_rate_threshold = success_rate_threshold
        # difficulty level
        self.difficulty = 1
        # whether the episode was successful
        self.is_success = False

        # Select a pair of qubits from the coupling map as an action
        self.action_space = spaces.Discrete(len(self.coupling_map.get_edges()))
        self.action_dict = {k: v for k, v in enumerate(
            self.coupling_map.get_edges())}
        # observation space for observations given to the model
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.image_shape,
            dtype=np.uint8,
        )

        # generate the identity matrix as the desired goal
        self.desired_goal = np.eye(
            self.num_qubits, dtype=self.observation_space.dtype)
        self.desired_goal = self.desired_goal[np.newaxis, :, :]

        # maximum number of steps in an episode
        # the upperbound is O(n^2 / log n)
        self.max_steps = self.num_qubits**2

        self.reset()

    def convert_to_image(self, state: np.ndarray) -> np.ndarray:
        """
        Convert to discrete space if needed.

        """
        assert state.dtype == np.bool_
        return (state * 255).astype(np.uint8)

    def _get_obs(self) -> np.ndarray:
        """
        Helper to create the observation.

        Returns:
            np.ndarray: The observation
        """
        return self.convert_to_image(self.state.copy())

    def step(self, action):
        control, target = self.action_dict[action]
        self.cnot_gates.append((control, target))

        # apply CNOT gate
        self.state[0, target, :] = np.logical_xor(
            self.state[0, target, :], self.state[0, control, :], dtype=np.bool_
        )
        self.num_cnots += 1

        # if the binary matrix is the identity matrix, it is the goal
        terminated = bool(
            (np.eye(self.num_qubits, dtype=np.bool_) ^ self.state).sum() == 0
        )
        self.is_success = terminated

        # compute reward
        reward = float(self._compute_reward(self.state))

        info = {"is_success": self.is_success}

        # truncate if it reaches max_steps
        if self.difficulty is not None:
            truncated = self.num_cnots >= min(self.difficulty, self.max_steps)
        else:
            truncated = self.num_cnots >= self.max_steps

        if self.wandb_log:
            wandb.log(
                {
                    "Reward": reward,
                }
            )

        return self._get_obs(), reward, terminated, truncated, info

    # def _compute_reward(self, achieved_goal: np.ndarray) -> np.float32:
    #     reward = 0

    #     desired_goal = np.eye(self.num_qubits, dtype=np.bool_)

    #     distance = (achieved_goal ^ desired_goal).sum()

    #     qc = self._make_quantum_circuit()

    #     # give large points if the goal is reached
    #     if distance == 0:
    #         reward += 1
    #     else:
    #         reward -= distance / self.num_qubits**2 * 0.01
    #     # subtract points for each CNOT gate
    #     self.difficulty = 0 if self.difficulty is None else self.difficulty
    #     reward -= (self.num_cnots *
    #                (0.008 + 0.007*np.tanh(self.difficulty))) * 0.1 + qc.depth() * 0.002

    #     return reward
    def _compute_reward(self, achieved_goal: np.ndarray) -> np.float32:
        """
        ポテンシャルベースの多目的ハイブリッド報酬関数（PB-MHRF）に基づいて報酬を計算します。

        この関数は、エージェントが「行動（CNOT適用）」を終え、
        状態が s (直前の状態) から s' (achieved_goal) に遷移した直後に呼び出されることを想定しています。

        R_total = R_base + R_shaping
        """

        # --- 1. ポテンシャル関数 Phi(s) と Phi(s') の計算 ---

        # Phi(s) = -W * HammingDistance(s, goal)
        # ゴール（距離0）でポテンシャルが最大値 0 になるように設計します。

        # 現在の状態 s' (achieved_goal) のハミング距離とポテンシャルを計算
        A = achieved_goal
        B = self.desired_goal
        current_distance = (A & B).sum() / (A | B).sum()
        potential_s_prime = - 0.1 * current_distance / self.num_qubits**2

        # 直前の状態 s のポテンシャルを取得 (self.previous_distance は前のステップで保存済)
        potential_s = - 0.1 * self.previous_distance / self.num_qubits**2

        # --- 2. 整形報酬 R_shaping の計算 (PBRS) ---

        # R_shaping = gamma * Phi(s') - Phi(s)
        #
        # もしゴールに近づいた場合 (current_distance < self.previous_distance):
        #   -> potential_s_prime > potential_s
        #   -> R_shaping は「正の報酬」となり、学習をガイドします。
        #
        # もしゴールから遠ざかった場合 (current_distance > self.previous_distance):
        #   -> potential_s_prime < potential_s
        #   -> R_shaping は「負の報酬（罰）」となり、その行動を抑制します。
        #
        shaping_reward = (0.99 * potential_s_prime) - potential_s

        # --- 3. 基本報酬 R_base の計算 ---

        # R_base = R_sparse + R_step_cost
        # R_base は「真の目的」（正しく、かつ最小ステップで）を定義します。

        # R_sparse (終端報酬): 「正しさ」の報酬
        # ゴールに到達した場合のみ与えられます。
        sparse_reward = 1.0 if current_distance == 0 else 0.0

        # R_step_cost (ステップコスト): 「効率性」のペナルティ
        # この関数が呼ばれた＝1回CNOTを実行した、と仮定し、コストを課します。
        # これが累積することで、エージェントは総CNOT数を最小化しようとします。
        step_cost = - (0.008 + 0.007 * np.tanh(self.difficulty)) 

        # --- 4. 総報酬の計算 ---
        # total_reward = (sparse_reward + step_cost * self.num_cnots) + shaping_reward
        total_reward = sparse_reward + step_cost

        # --- 5. 次のステップのための状態更新 ---
        # 現在の距離を「直前の距離」として保存します。
        self.previous_distance = current_distance

        return np.float32(total_reward)

    def _make_quantum_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        try:
            for control, target in self.cnot_gates:
                qc.cx(control, target)
        except:
            pass
        return None

    def reset(self, seed=None, options=None):
        # raise difficulty level when success rate is high
        self._update_difficulty()

        if self.wandb_log:
            wandb.log(
                {
                    "CNOT count": self.num_cnots,
                    "depth": self._make_quantum_circuit().depth()
                }
            )

        self.state = linear_function_circuit_to_binary_matrix(
            random_linear_function_circuit_by_difficulty(
                self.num_qubits, self.difficulty, self.coupling_map
            )
        )
        self.state = self.state[np.newaxis, :, :]
        self.is_success = False
        self.num_cnots = 0
        self.cnot_gates = []

        # 直前のステップのハミング距離を保持する変数
        # reset()時に、初期状態の距離で初期化する必要があります
        A = self.state
        B = self.desired_goal
        self.previous_distance = (A & B).sum() / (A | B).sum()

        return self._get_obs(), {}

    def _update_difficulty(self):
        self.total_count += 1
        if self.is_success:
            self.success_count += 1

        if self.total_count == self.eval_batch_size:
            success_rate = self.success_count / self.total_count
            if self.difficulty < self.max_steps:
                if success_rate > self.success_rate_threshold:
                    self.difficulty += 1
            self.total_count = 0
            self.success_count = 0

            if self.wandb_log:
                wandb.log(
                    {
                        "Difficulty": self.difficulty,
                        "Success rate": success_rate,
                    }
                )

    def render(self):
        qc = QuantumCircuit(self.num_qubits)
        for control, target in self.cnot_gates:
            qc.cx(control, target)
        print("QuantumCircuit:")
        print(qc)
        print(f"state:\n{self.state}")
        print(f"num_cnots: {self.num_cnots}")
        return None

    def close(self):
        pass


class AILinearFunctionSynthesisNoCurriculumLearning(AILinearFunctionSynthesis):
    """Curriculum learning disabled environment for AI linear function synthesis"""

    def _update_difficulty(self):
        self.difficulty = None

        self.total_count += 1
        if self.is_success:
            self.success_count += 1

        if self.total_count == self.eval_batch_size:
            success_rate = self.success_count / self.total_count
            self.total_count = 0
            self.success_count = 0

            if self.wandb_log:
                wandb.log(
                    {
                        "Success rate": success_rate,
                    }
                )


class AILinearFunctionSynthesisDenseReward(AILinearFunctionSynthesis):
    """Environment with dense rewards for AI linear function synthesis"""

    def _compute_reward(self, achieved_goal: np.ndarray) -> np.float32:
        reward = 0

        desired_goal = np.eye(self.num_qubits, dtype=np.bool_)

        distance = (achieved_goal ^ desired_goal).sum()

        # give large points if the goal is close
        reward -= distance
        # subtract points for each CNOT gate
        reward -= self.num_cnots * 0.01
        return reward
        return reward
        return reward
        return reward
