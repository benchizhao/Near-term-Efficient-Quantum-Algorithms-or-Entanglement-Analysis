# README

This directory contains 4 files corresponding to 4 experiments, respectively, of the paper **VARIATIONAL QUANTUM ALGORITHMS FOR SCHMIDT DECOMPOSITION**. The first 2 experiments are based on *Paddle Quantum* platform and the rest are based on *Quantum Leaf* platform.

### Paddle Quantum
Paddle Quantum (量桨) is a quantum machine learning (QML) toolkit developed based on Baidu PaddlePaddle. It provides a platform to construct and train quantum neural networks (QNNs) with easy-to-use QML development kits supporting combinatorial optimization, quantum chemistry and other cutting-edge quantum applications, making PaddlePaddle the first and only deep learning framework in China that supports quantum machine learning.

Paddle Quantum could be installed via the following command:

`pip install paddle-quantum==2.1.0`

For more information, please refer to [Paddle Quantum](https://qml.baidu.com/).

### Quantum Leaf
Quantum Leaf (量易伏) is a Cloud-Native quantum computing platform developed by the Institute for Quantum Computing, Baidu. It is used for programming, simulating and executing quantum computers, aiming at providing the quantum programming environment for Quantum infrastructure as a Service (QaaS).

QCompute is a Python-based open-source SDK. It provides a full-stack programming experience for advanced users via the features of hybrid quantum programming language and a high-performance simulator. Users can use the already-built objects and modules of quantum programming environment, pass parameters to build and execute the quantum circuits on the local simulator or the cloud simulator/hardware.

**Quantum Leaf has access to 10-qubit superconducting Quantum Device in Institute of Physics, Chinese Academy of Sciences (IoPCAS)**

QCompute could be installed via the following command:

`pip install qcompute==2.0.0`

For more information, please refer to [Quantum Leaf](https://quantum-hub.baidu.com/services).

> **Note:** To run the codes properly, put the folder qapp in the same path of the script files. The folder qapp can be found in QCompute/Example/QAPP.

### Packages Version
- python = 3.8.13
- paddle-quantum = 2.2.0
- paddlepaddle = 2.3.0
- qcompute = 2.0.0
- numpy = 1.19.3
- matplotlib = 3.5.1

### File Description
The structure of each file are similar:

`exp(i).py` defines the setups and optimization procedures or each task. All data achieved will be save in *data_(i)* file.
`figure(i).py` reads the data from *data_(i)* file and generate plots, which is saved at *this file*.
`vqasd` provides helper functions for experiments.

All experiment data is save already, figures can be plotted directed.

> **Note:** In file **exp3-implementation of superconducting quantum processor**, users are suppose to register with [Quantum Leaf](https://quantum-hub.baidu.com/services) to obtain personal **Token**. Otherwise, user has no access to the cloud services, including quantum device.