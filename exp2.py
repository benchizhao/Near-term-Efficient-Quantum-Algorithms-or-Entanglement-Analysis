import numpy as np
import paddle
import os
import paddle_quantum
from vqasd import VQASD2



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

paddle_quantum.set_backend('density_matrix')

num_qubit = 2
DEPTH = 1
d = 2 ** int(num_qubit / 2)

# type of noise, set to be 'depolarizing' or 'amplitude_damping'
noise_type = 'depolarizing'
# noise_type = 'amplitude_damping'
# 
coeff_list = []
p_vals = np.linspace(0, 1, 11)
for i in range(len(p_vals)):
    p = p_vals[i]
    ITR = 100  # number of iterations
    LR = 0.05  # learing rate
    print('Simulating ' + noise_type + ' noise with intensity ' + str(np.around(p, decimals=1)) + '...')
    net = VQASD2(noise_type, p)
    # choose Adam optmizer
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())
    plot_list = []

    # optmizing interations
    for itr in range(ITR):
        loss = net.loss_compute()
        loss.backward()  # back propogation
        opt.minimize(loss)
        opt.clear_grad()  # clear gradient
        coeff = net.coefficient_compute()
        distance = sum((np.array(coeff) - 1 / np.sqrt(d)) ** 2)
        plot_list.append(distance)
        # print results
        if itr % 20 == 0:
            print("  itr " + str(itr) + " of", ITR)
    coeff_list.append(net.coefficient_compute())
    print('(Estimated) first', d, "Schmidt Coefficients:\n", net.coefficient_compute())
    print("distance:", plot_list[-1])

# save files for amplitude damping channel
filename = './data_2/Schmidt_coeff_with_Depolarizing_channel_p=[0,1].npz'
if noise_type == 'amplitude_damping':
    # save files for depolarizing channel
    filename = './data_2/Schmidt_coeff_with_AD_channel_p=[0,1].npz'
# os.makedirs("data_2", exist_ok=True)
# np.savez(filename, variable=p_vals, coeff=coeff_list)
