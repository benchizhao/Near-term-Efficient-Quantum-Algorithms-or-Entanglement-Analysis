import numpy as np
from QCompute.QPlatform.QOperation.RotationGate import RY
from QCompute.QPlatform.QOperation.FixedGate import CZ
from QCompute.QPlatform import BackendName
from qapp.circuit import ParameterizedCircuit
from qapp.optimizer import SMO
from vqasd import VQASD4
import os

from QCompute import Define

Define.Settings.outputInfo = False


class OneSideEntangledCircuit(ParameterizedCircuit):
    def __init__(self, num: int, layer: int, parameters: np.ndarray):
        """The constructor of the one-side parameterized circuit class

        :param num: Number of qubits in this Ansatz
        :param layer: Number of layer for this Ansatz
        :param parameters: Parameters of parameterized gates in the parameterized circuit (len: num * layer * 3)
        """
        super().__init__(num, parameters)
        self._layer = layer

    def add_circuit(self, q):
        """Adds parameterized circuit to the register, note that the circuit is one-side

        :param q: Quantum register to which this circuit is added
        """
        CZ(q[0], q[1])
        CZ(q[1], q[2])
        for i in range(self._layer):
            for j in range(int(self._num / 2)):
                RY(self._parameters[i * self._num + j])(q[j])
            CZ(q[1], q[2])
            CZ(q[0], q[1])


layer = 3
num_qubits = 6
num_sample = 10
rank_max = 8
ITR = 8
shots = 10000

for rank in range(1):
    rank = 7
    print('\nSampling rank-%d states...' % rank)
    L1_all = []
    for sample in range(num_sample):
        print('\n  Estimating Log. Neg. of %d-th state...' % (sample + 1))
        parameters = 2 * np.pi * np.random.rand(num_qubits * layer) - np.pi
        ansatz = OneSideEntangledCircuit(num_qubits, layer, parameters)
        opt = SMO(ITR, ansatz)
        vsd = VQASD4(rank, num_qubits, ansatz, opt, BackendName.LocalBaiduSim2)

        vsd.run(shots)

        L1norm = np.sqrt(-opt._loss_history[-1] * (2 ** (num_qubits / 2)))
        L1_all.append(L1norm)
        # os.makedirs("data_4", exist_ok=True)
        # np.save("./data_4/rank=" + str(rank), L1_all)
