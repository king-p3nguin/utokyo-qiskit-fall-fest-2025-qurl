from gymnasium.envs.registration import register
from qiskit.transpiler import CouplingMap
from stable_baselines3.common.env_checker import check_env

from .circuit import *
from .env import *

TEST_COUPLING_MAP = CouplingMap.from_line(6)
TEST_BATCH_SIZE = 100
TEST_THRESHOLD = 0.8

env_kwargs = {
    "coupling_map": TEST_COUPLING_MAP,
    "eval_batch_size": TEST_BATCH_SIZE,
    "success_rate_threshold": TEST_THRESHOLD,
}
check_env(AILinearFunctionSynthesis(**env_kwargs))
check_env(AILinearFunctionSynthesisNoCurriculumLearning(**env_kwargs))
check_env(AILinearFunctionSynthesisDenseReward(**env_kwargs))

register(
    id="AILinearFunctionSynthesis-v0",
    entry_point="ai_linear_function_synthesis.env:AILinearFunctionSynthesis",
)
