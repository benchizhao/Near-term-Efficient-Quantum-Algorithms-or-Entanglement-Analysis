import numpy as np
import paddle
import os
from vqasd import VQASD1


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

num_qubit = 8
DEPTH = 1  # Depth of circuits, set to be 1, 2, 4, 8 separately
d = 2 ** int(num_qubit / 2)

ITR = 120  # number of iterations
LR = 0.2  # learning rate
net = VQASD1(DEPTH)
# chose Adam optimizer
opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())
loss_list = []
plot_list = []
print("Optimizing circuit with %d layer(s)" % DEPTH)
# optimizing iterations
for itr in range(ITR):
    loss = net.loss_compute()
    # back propogate
    loss.backward()
    opt.minimize(loss)
    # clear gradient
    opt.clear_grad()
    loss_list.append(loss.numpy()[0])
    coeff = net.coefficient_compute()
    distance = sum((np.array(coeff) - 1 / np.sqrt(d)) ** 2)
    plot_list.append(distance)
    # print results
    if itr % 5 == 0:
        print("itr " + str(itr) + ", loss =", loss.numpy()[0])
        print('Distance between real and estimated Coefficients:', distance)

os.makedirs("data_1", exist_ok=True)
# filename = './data_1/data_of_' + str(num_qubit) + '_qubits_with_' + str(DEPTH) + '_depth.npz'
# np.savez(filename, error=plot_list, loss=loss_list, eigenval=net.coefficient_compute())
