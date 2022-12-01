import numpy as np
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
all_sim = []
all_real = []
coeff_sim = []
coeff_real = []
for i in range(20):
    filename_sim = './data_3/Simulator_data_loss_list_time=' + str(i) + '.npz'
    data_sim = np.load(filename_sim)
    loss_sim = -data_sim['loss']
    all_sim.append(loss_sim)
    coeff_sim.append(data_sim['coeff'])

    filename_real = './data_3/Quantum_Device_data_loss_list_time=' + str(i) + '.npz'
    data_real = np.load(filename_real)
    loss_real = -data_real['loss']
    all_real.append(loss_real)
    coeff_real.append(data_real['coeff'])

all_sim = np.array(all_sim)
all_real = np.array(all_real)
# print(all_real[:, 1])

ave_sim = []
ave_real = []
for i in range(10):
    ave_sim.append(sum(all_sim[:, i]) / len(all_sim[:, i]))
    ave_real.append(sum(all_real[:, i]) / len(all_real[:, i]))

std_sim = []
std_real = []
for i in range(10):
    std_sim.append(np.std(all_sim[:, i]))
    std_real.append(np.std(all_real[:, i]))

max_sim = []
min_sim = []
max_real = []
min_real = []
for i in range(10):
    max_sim.append(max(all_sim[:, i]))
    min_sim.append(min(all_sim[:, i]))
    max_real.append(max(all_real[:, i]))
    min_real.append(min(all_real[:, i]))

fig = plt.figure(0)
plt.xlabel('Iterations')
plt.ylabel('Cost')

plt.plot(ave_sim, '-o', label='Simulator')
plt.plot(ave_real, '-^', label='Quantum Device')
plt.fill_between(x, min_sim, max_sim, alpha=0.3)
plt.fill_between(x, min_real, max_real, alpha=0.3)

plt.legend(loc='best')
# plt.savefig('fig3.pdf')
plt.show()

ave_coeff_sim = np.sum(np.array(coeff_sim), axis=0) / len(coeff_real)
ave_coeff_sim_std = np.std(np.array(coeff_sim), axis=0)

ave_coeff_real = np.sum(np.array(coeff_real), axis=0) / len(coeff_real)
ave_coeff_real_std = np.std(np.array(coeff_real), axis=0)

print('mean of simulation coefficient=', ave_coeff_sim)
print('std of simulation coefficient=', ave_coeff_sim_std)
print('mean of quantum device coefficient=', ave_coeff_real)
print('std of quantum device coefficient=', ave_coeff_real_std)
