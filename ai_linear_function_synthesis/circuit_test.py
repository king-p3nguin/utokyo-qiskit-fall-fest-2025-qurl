import pytest
from qiskit.transpiler import CouplingMap

from .circuit import *


def test_linear_function_circuit_to_binary_matrix():
    num_qubits = 8
    qc = random_linear_function_circuit(8)
    matrix = linear_function_circuit_to_binary_matrix(qc)
    assert gf2_rank(matrix) == num_qubits


@pytest.mark.parametrize(
    "num_qubits, difficulty, coupling_map",
    [
        (8, None, CouplingMap.from_line(8)),
        (8, 1, CouplingMap.from_line(8)),
        (8, 5, CouplingMap.from_line(8)),
        (8, 10, CouplingMap.from_line(8)),
        (8, 15, CouplingMap.from_line(8)),
    ],
)
def test_random_linear_function_circuit_by_difficulty(
    num_qubits, difficulty, coupling_map
):
    qc = random_linear_function_circuit_by_difficulty(
        num_qubits, difficulty, coupling_map
    )
    matrix = linear_function_circuit_to_binary_matrix(qc)
    assert gf2_rank(matrix) == num_qubits
