import numpy as np
import paddle_quantum
from paddle import matmul, trace, sqrt, real
from paddle_quantum.ansatz import Circuit
import paddle
from QCompute.QPlatform.QEnv import QEnv
from QCompute.QPlatform.QOperation.FixedGate import CZ, CX, H, X, CH, CCX
from QCompute.QPlatform.QOperation.RotationGate import RY
from QCompute.QPlatform.QOperation.Measure import MeasureZ
from scipy.stats import unitary_group
from paddle_quantum.state.common import bell_state, zero_state, to_state
from qapp.circuit import ParameterizedCircuit
from qapp.optimizer import BasicOptimizer
from typing import Union

def state_preparation(n, p):
    """
    Entangled state preparation with respect to Schmidt coefficients s
    :param n: number of qubits
    :param p: the coefficient square vector summing up to 1
    :return: the state vector with the form of sum_j p_j|j>|j>
    """
    if abs(sum(p) - 1) > 10e-5:
        print('Error s format')
        return 0
    rho = np.eye(2 ** int(n / 2)) / 2 ** int(n / 2)
    eigenvalue, featurevector = np.linalg.eig(rho)
    phi = np.sqrt(p[0]) * np.kron(featurevector[:, 0], featurevector[:, 0])
    for j in range(1, len(p)):
        phi += np.sqrt(p[j]) * np.kron(featurevector[:, j], featurevector[:, j])

    return phi


class VQASD1(paddle.nn.Layer):
    def __init__(self, depth):
        super(VQASD1, self).__init__()
        self._n = 8
        self._depth = depth
        self.initial_state = to_state(self.init_state())
        self.cir = self.U_theta(self._n, self._depth)
        p_list = []
        p0 = 0.1
        for _ in range(2 ** int(self._n / 2)):
            p_list.append(p0 + _)
        p_list = p_list[::-1]
        p_list = np.array(p_list) / sum(p_list)
        psi = state_preparation(self._n, p_list)
        self.measure_state = paddle.to_tensor(np.outer(psi, psi), dtype='complex64')

    def init_state(self):
        paddle_quantum.set_backend('density_matrix')
        U1 = unitary_group.rvs(16)
        U2 = unitary_group.rvs(16)
        U = np.kron(U1, U2)
        state = bell_state(8).numpy()
        fin_state = U @ state @ U.conj().T
        return paddle.to_tensor(fin_state, dtype='complex64')
    
    def U_theta(self, num_qubit:int, depth:int)->Circuit:
        cir = Circuit(num_qubit)
        cir.complex_entangled_layer([0, 1, 2, 3], 8, depth)
        cir.complex_entangled_layer([4, 5, 6, 7], 8, depth)
        return cir
    
    def loss_compute(self):
        rho_middle = self.cir(self.initial_state).data
        # rho_middle = paddle.to_tensor(rho_middle.numpy(), dtype='complex64')
        fid = -real(trace(matmul(rho_middle, self.measure_state)))
        return fid

    def coefficient_compute(self):
        coefficient_list = []

        d = 2 ** int(self._n / 2)
        for j in range(d):
            e_j = np.kron(np.eye(d)[:, j], np.eye(d)[:, j])
            rho_j = paddle.to_tensor(np.outer(e_j, e_j), dtype='complex64')
            rho_out = paddle.to_tensor(self.cir(self.initial_state).numpy(), dtype='complex64')
            fid = sqrt(real(trace(matmul(rho_out, rho_j))))
            coefficient_list.append(fid.numpy()[0])

        return coefficient_list


class VQASD2(paddle.nn.Layer):
    def __init__(self, noise_type, p):
        super(VQASD2, self).__init__()
        self._n = 2
        self._noise_type = noise_type
        self.initial_state = self.init_state()
        self.noise_state = self.init_state_noise(p)
        self.cir = self.U_theta(self._n)
        self.measure_state = zero_state(2)
    
    def init_state(self):
        init_s = zero_state(2)
        theta_1 = paddle.to_tensor(np.array([0.58]), dtype='complex64')
        theta_2 = paddle.to_tensor(np.array([1.58]), dtype='complex64')

        cir = Circuit(2)
        cir.ry(qubits_idx=[0], param=theta_1)
        cir.ry(qubits_idx=[1], param=theta_2)
        cir.h(1)
        cir.cnot([0, 1])
        cir.h(1)

        fin_state = cir(init_s)
        return fin_state
    
    def init_state_noise(self, p):
        init_s = zero_state(2)
        theta_1 = paddle.to_tensor(np.array([0.58]), dtype='complex64')
        theta_2 = paddle.to_tensor(np.array([1.58]), dtype='complex64')

        cir = Circuit(2)
        cir.ry(qubits_idx=[0], param=theta_1)
        cir.ry(qubits_idx=[1], param=theta_2)
        cir.h(1)
        cir.cnot([0, 1])
        cir.h(1)

        if self._noise_type == 'depolarizing':
            # depolarizing channel
            cir.depolarizing(p, [0])
            cir.depolarizing(p, [1])
        elif self._noise_type == 'amplitude_damping':
            # amplitude damping channel
            cir.amplitude_damping(p, [0])
            cir.amplitude_damping(p, [1])
        # fix theta
        cir_theta = cir.parameters()
        for t in cir_theta:
            t.stop_gradient = True
    
        final_state = cir(init_s)
        return final_state
    
    def U_theta(self, num_qubit:int)->Circuit:
        cir = Circuit(num_qubit)
        cir.u3(qubits_idx=0)
        cir.u3(qubits_idx=1)
        return cir
    
    def loss_compute(self):
        rho_middle = self.cir(self.noise_state)
        fid = -real(trace(matmul(rho_middle.data, self.measure_state.data)))
        return fid

    def coefficient_compute(self):
        coefficient_list = []

        d = 2 ** int(self._n / 2)
        for j in range(d):
            e_j = np.kron(np.eye(d)[:, j], np.eye(d)[:, j])
            rho_j = paddle.to_tensor(np.outer(e_j, e_j), dtype='complex64')
            rho_out = self.cir(self.initial_state)
            fid = sqrt(real(trace(matmul(rho_out.data, rho_j))))
            coefficient_list.append(fid.numpy()[0])

        return coefficient_list


class VQASD3:
    """
    Variational Schmidt Decomposition class
    Specified for Exp 3
    """

    def __init__(self, num: int, ansatz: 'ParameterizedCircuit',
                 optimizer: 'BasicOptimizer', backend: str):
        self._num = num
        self._ansatz = ansatz
        self._optimizer = optimizer
        self._backend = backend
        self._schmidt_coefficients = "Run VSD.run() first"

    def _input_state_preparation(self, q):
        """Prepare input state
        """
        RY(0.58)(q[0])
        RY(1.58)(q[1])
        CZ(q[0], q[1])

    def _compute_loss(self, parameters: np.ndarray, shots: int) -> float:
        """Compute loss
        """
        self._ansatz.set_parameters(parameters)
        env = QEnv()
        env.backend(self._backend)
        q = env.Q.createList(self._num)

        self._input_state_preparation(q)

        # Add PQC
        self._ansatz.add_circuit(q[:self._num])

        # Measurement
        MeasureZ(q, list(range(self._num)))
        # Submit job
        counts = env.commit(shots, fetchMeasure=True)['counts']
        # Expectation
        result = (counts.get('00', 0)) / shots

        return - result

    def _compute_gradient(self, parameters: np.ndarray, shots: int) -> np.ndarray:
        """Compute gradient
        """
        gradient = np.zeros_like(parameters)
        for i in range(len(parameters)):
            param_plus = parameters.copy()
            param_minus = parameters.copy()
            param_plus[i] += np.pi / 2
            param_minus[i] -= np.pi / 2
            loss_plus = self._compute_loss(param_plus, shots)
            loss_minus = self._compute_loss(param_minus, shots)
            gradient[i] = ((loss_plus - loss_minus) / 2)
        self._ansatz.set_parameters(parameters)

        return gradient

    def _coefficient_read_out(self, shots: int = 1024):
        env = QEnv()
        env.backend(self._backend)
        q = env.Q.createList(self._num)

        self._input_state_preparation(q)

        # Add circuit
        self._ansatz.add_circuit(q[:self._num])

        # Measurement
        MeasureZ(q, list(range(self._num)))
        # Submit job
        counts = env.commit(shots, fetchMeasure=True)['counts']
        # Expectation
        self._schmidt_coefficients = []
        self._schmidt_coefficients.append(np.sqrt((counts.get('00', 0)) / shots))
        self._schmidt_coefficients.append(np.sqrt((counts.get('11', 0)) / shots))

    def run(self, shots: int = 1024):
        self._optimizer.minimize(shots, self._compute_loss, self._compute_gradient)
        self._coefficient_read_out(shots)

    @property
    def schmidt_coefficients(self) -> Union[str, np.ndarray]:
        """The optimized eigenvalue from last run

        :return: The optimized eigenvalue from last run
        """

        return self._schmidt_coefficients


class VQASD4:
    """
    Variational Schmidt Decomposition class
    Specified for Exp 4
    """

    def __init__(self, rank: int, num: int, ansatz: 'ParameterizedCircuit',
                 optimizer: 'BasicOptimizer', backend: str):
        self._num = num
        self._rank = rank
        self._ansatz = ansatz
        self._optimizer = optimizer
        self._backend = backend
        self._random_input = np.random.rand(num * 2)

    def _input_state_preparation(self, r, q):
        # equal amplitudes
        if r == 1:
            pass
        elif r == 2:
            H(q[2])
        elif r == 3:
            RY(1.230959)(q[1])
            X(q[1])
            CH(q[1], q[2])
            X(q[1])
        elif r == 4:
            H(q[1])
            H(q[2])
        elif r == 5:
            RY(0.927295)(q[0])
            X(q[0])
            CH(q[0], q[1])
            CH(q[0], q[2])
            X(q[0])
        elif r == 6:
            RY(1.230959)(q[0])
            X(q[0])
            CH(q[0], q[1])
            X(q[0])
            H(q[2])
        elif r == 7:
            RY(1.4)(q[0])
            RY(1.4)(q[1])
            H(q[2])
            RY(np.pi / 4)(q[2])
            CCX(q[0], q[1], q[2])
            RY(-np.pi / 4)(q[2])
        else:
            H(q[0])
            H(q[1])
            H(q[2])

        for j in range(int(self._num / 2)):
            CX(q[j], q[j + int(self._num / 2)])

        # some random unitary
        for j in range(self._num):
            RY(self._random_input[j])(q[j])
        CZ(q[0], q[1])
        CZ(q[1], q[2])
        CZ(q[2], q[0])
        CZ(q[3], q[4])
        CZ(q[4], q[5])
        CZ(q[5], q[3])
        for j in range(self._num):
            RY(self._random_input[j + self._num])(q[j])
        CZ(q[0], q[1])
        CZ(q[1], q[2])
        CZ(q[2], q[0])
        CZ(q[3], q[4])
        CZ(q[4], q[5])
        CZ(q[5], q[3])

    def _w_dagger(self, q):
        # Entanglement weighting
        for j in range(int(self._num / 2)):
            CX(q[j], q[j + int(self._num / 2)])
            H(q[j])

    def _compute_loss(self, parameters: np.ndarray, shots: int) -> float:
        """Compute loss
        """
        self._ansatz.set_parameters(parameters)
        env = QEnv()
        env.backend(self._backend)
        q = env.Q.createList(self._num)

        # prepare input
        self._input_state_preparation(self._rank, q)

        # Add circuit
        self._ansatz.add_circuit(q)

        # Entanglement weighting
        self._w_dagger(q)

        # Measurement
        MeasureZ(q, list(range(self._num)))

        counts = env.commit(shots, fetchMeasure=True)['counts']
        # Expectation
        result = (counts.get('000000', 0)) / shots

        return - result

    def _compute_gradient(self, parameters: np.ndarray, shots: int) -> np.ndarray:
        """Compute gradient
        """
        gradient = np.zeros_like(parameters)
        for i in range(len(parameters)):
            param_plus = parameters.copy()
            param_minus = parameters.copy()
            param_plus[i] += np.pi / 2
            param_minus[i] -= np.pi / 2
            loss_plus = self._compute_loss(param_plus, shots)
            loss_minus = self._compute_loss(param_minus, shots)
            gradient[i] = ((loss_plus - loss_minus) / 2)
        self._ansatz.set_parameters(parameters)

        return gradient

    def run(self, shots: int = 1024):
        """Searches the minimum eigenvalue of the input Hamiltonian with the given ansatz and optimizer.

        :param shots: Number of measurement shots, defaults to 1024
        """
        self._optimizer.minimize(shots, self._compute_loss, self._compute_gradient)

    def set_backend(self, backend: str):
        """Sets the backend to be used

        :param backend: The backend to be used
        """
        self._backend = backend
