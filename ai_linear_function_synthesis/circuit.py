import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import LinearFunction
from qiskit.transpiler import CouplingMap


def gf2_rank(matrix: list) -> int:
    """Calculate the rank of a binary matrix in GF(2).
    Ref: https://gist.github.com/StuartGordonReid/eb59113cb29e529b8105?permalink_comment_id=3268301#gistcomment-3268301

    Args:
        matrix: A list of lists representing a binary matrix.

    Returns:
        The rank of the matrix.
    """
    matrix = list(matrix)
    n = len(matrix[0])
    rank = 0
    for col in range(n):
        j = 0
        rows = []
        while j < len(matrix):
            if matrix[j][col] == 1:
                rows += [j]
            j += 1
        if len(rows) >= 1:
            for c in range(1, len(rows)):
                for k in range(n):
                    matrix[rows[c]][k] = (matrix[rows[c]][k] + matrix[rows[0]][k]) % 2
            matrix.pop(rows[0])
            rank += 1
    for row in matrix:
        if sum(row) > 0:
            rank += 1
    return rank


def random_invertible_binary_matrix(n: int) -> np.ndarray:
    rank = 0
    while rank != n:
        mat = np.random.randint(0, 2, (n, n))
        rank = gf2_rank(mat)
    return mat


def random_linear_function_circuit(num_qubits: int) -> QuantumCircuit:
    matrix = random_invertible_binary_matrix(num_qubits)
    circuit = QuantumCircuit(num_qubits)
    circuit.append(LinearFunction(matrix), circuit.qubits)
    return circuit.decompose()


def linear_function_circuit_to_binary_matrix(circuit: QuantumCircuit) -> np.ndarray:
    """Convert a quantum circuit implementing a linear function to a binary matrix.
    Ref: https://github.com/Qiskit/qiskit/blob/a0918b8fc6c25bd9cb35ede1b7778f7ae9178aed/qiskit/circuit/library/generalized_gates/linear_function.py#L152-L187

    Args:
        circuit: The quantum circuit implementing a linear function.

    Returns:
        The binary matrix corresponding to the linear function.
    """
    num_qubits = circuit.num_qubits
    mat = np.eye(num_qubits, num_qubits, dtype=np.bool_)

    for instruction in circuit.data:
        if instruction.operation.name in ("barrier", "delay"):
            # can be ignored
            pass
        elif instruction.operation.name == "cx":
            # implemented directly
            cb = circuit.find_bit(instruction.qubits[0]).index
            tb = circuit.find_bit(instruction.qubits[1]).index
            mat[tb, :] = (mat[tb, :]) ^ (mat[cb, :])
        elif instruction.operation.name == "swap":
            # implemented directly
            cb = circuit.find_bit(instruction.qubits[0]).index
            tb = circuit.find_bit(instruction.qubits[1]).index
            mat[[cb, tb]] = mat[[tb, cb]]
        else:
            raise ValueError(f"Unsupported gate: {instruction.operation.name}")

    return mat


def random_linear_function_circuit_by_difficulty(
    num_qubits: int, difficulty: int | None, coupling_map: CouplingMap
) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    edges = coupling_map.get_edges()
    num_choices = len(edges)
    if difficulty is not None:
        for _ in range(difficulty):
            q0, q1 = edges[np.random.choice(num_choices, replace=False)]
            circuit.cx(q0, q1)
    else:
        circuit = random_linear_function_circuit(num_qubits)
    return circuit
