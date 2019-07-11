import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


def func(x, k1, k2, k3):
    return k1 + k2 * np.exp(-x/k3)


if __name__ == "__main__":
    x = np.loadtxt('x.txt')
    y = np.loadtxt('y.txt')
    p0 = [y.max(), y.max() - y.min(), x[-1] / 3]

    fit = curve_fit(func, x, y, p0=p0)

    plt.plot(x, y)
    plt.plot(x, func(x, *fit[0]))
    plt.show()
