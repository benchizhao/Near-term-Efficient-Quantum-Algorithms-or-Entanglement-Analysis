from matplotlib import pyplot as plt
import numpy as np

mk = ["<", "o", "*", "v", "+", ".", "x", "d", "p", ">", "^", "s"]
ls = [":", "-.", "--", (0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1)), ":", "-.", "--", "-", ":", "-.", "--", "-"]
clr = ["y", "g", "b", "m", "r", "c", "pink", "teal", "navy", "peru", "purple", "k"]


def plot_fig4():
    k = 1
    boxdata = np.zeros((10, 1))
    linedata = []
    for rank in range(1, 9):
        data = np.load("./data_4/rank=" + str(rank) + '.npy')
        data = 2 * np.log2(data)
        linedata.append(np.median(data))
        boxdata = np.c_[boxdata, data]
    boxdata = np.delete(boxdata, 0, axis=1)
    plt.boxplot(boxdata, labels=range(1, 9), whis=1.0, sym='', patch_artist=True, widths=0.25,
                capprops={'color': clr[k]}, medianprops={'color': "r"}, whiskerprops={'color': clr[k]},
                boxprops={'color': clr[k], 'facecolor': clr[k]})
    plt.plot(range(1, 9), linedata, color=clr[k], linestyle=ls[-1], marker='',
             label='estimated')
    # plot
    value = []
    for rank in range(1, 9):
        value.append(rank * np.sqrt(1 / (rank)))
    value = 2 * np.log2(value)
    plt.plot(range(1, 9), value, marker='>', label='theoretical')

    plt.xlabel('Schmidt Ranks of ' + r'$|\psi\rangle$', fontsize=14)
    plt.ylabel('Logarithm Negativity', fontsize=14)
    plt.legend(fontsize=12)
    # plt.savefig('fig4.pdf')
    plt.show()


plot_fig4()
