import numpy as np
import matplotlib.pyplot as plt

filename_AD = './data_2/Schmidt_coeff_with_AD_channel_p=[0,1].npz'
data_AD_channel = np.load(filename_AD)
variable_AD = data_AD_channel['variable']
coeff_list_AD = np.array(data_AD_channel['coeff'])
coeff_list_AD = np.sort(coeff_list_AD, axis=1)


filename_DE = './data_2/Schmidt_coeff_with_Depolarizing_channel_p=[0,1].npz'
data_DE_channel = np.load(filename_DE)
variable_DE = data_DE_channel['variable']
coeff_list_DE = np.array(data_DE_channel['coeff'])
coeff_list_DE = np.sort(coeff_list_DE, axis=1)

fig = plt.figure(0)
plt.xlabel('Noise level p')
plt.ylabel('Schmidt coefficients')

plt.plot(variable_AD, [0.958]*len(coeff_list_AD), 'k--')
plt.plot(variable_AD, [0.286]*len(coeff_list_AD), 'k--')

plt.plot(variable_AD[0:-1], coeff_list_AD[:, 0][0:-1], 'bo', label='1st Schm. coeff. of AD noised state')
plt.plot(variable_AD[0:-1], coeff_list_AD[:, 1][0:-1], 'b^', label='2nd Schm. coeff. of AD noised state')

plt.plot(variable_DE[0:-1], coeff_list_DE[:, 0][0:-1], 'ro', label=r'1st Schm. coeff. of depolarizing noised state')
plt.plot(variable_DE[0:-1], coeff_list_DE[:, 1][0:-1], 'r^', label=r'2st Schm. coeff. of depolarizing noised state')


plt.legend(loc='best')
# plt.savefig(fname="fig2.pdf")
plt.show()
