import numpy as np
import matplotlib.pyplot as plt

data1 = np.load('./data_1/data_of_8_qubits_with_1_depth.npz')
data2 = np.load('./data_1/data_of_8_qubits_with_2_depth.npz')
data3 = np.load('./data_1/data_of_8_qubits_with_4_depth.npz')
data4 = np.load('./data_1/data_of_8_qubits_with_8_depth.npz')

error_1 = data1['error']
error_2 = data2['error']
error_3 = data3['error']
error_4 = data4['error']

x = np.arange(0, 100, 5)

y_1 = [error_1[i] for i in x]
y_2 = [error_2[i] for i in x]
y_3 = [error_3[i] for i in x]
y_4 = [error_4[i] for i in x]

fig = plt.figure(0)
plt.xlabel('ITR')
plt.ylabel('Error')

plt.plot(x, y_1, '-b^', label='DEPTH=1')
plt.plot(x, y_2, '-go', label='DEPTH=2')
plt.plot(x, y_3, '-rv', label='DEPTH=4')
plt.plot(x, y_4, '-c*', label='DEPTH=8')


plt.legend(loc='best')
# plt.savefig(fname="fig1.pdf")
plt.show()
