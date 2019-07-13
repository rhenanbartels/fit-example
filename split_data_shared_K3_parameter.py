import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


x = np.loadtxt('x.txt')
y = np.loadtxt('y.txt')

# for later code clarity, be specific
allX = x
allY = y

splitN = 10
# calculate the overlapping split fraction of data
# split off the overlapping data segments for y
yLow = y[:splitN]
yHigh = y[splitN:]

# split off the overlapping data segments for x
xLow = x[:splitN]
xHigh = x[splitN:]


def funcLow(data, k1_lo, k2_lo, k3_shared):
        return k1_lo + k2_lo * np.exp(-data/k3_shared)

def funcHigh(data, k1_hi, k2_hi, k3_shared):
        return k1_hi + k2_hi * np.exp(-data/k3_shared)


# scipy's curve_fit takes a single function with a single data reference
def combinedFunction(comboData, k1_lo,k1_hi, k2_lo, k2_hi, k3_shared):
    # single data reference passed in, extract separate data
    extract1 = comboData[:splitN] # low data
    extract2 = comboData[splitN:] # high data
    
    result1 = funcLow(extract1, k1_lo, k2_lo, k3_shared)
    result2 = funcHigh(extract2, k1_hi, k2_hi, k3_shared)

    # scipy will only compare a single result, combine the seperate results
    return np.append(result1, result2)


# some initial parameter values for the combined function, so
# these are for k1_lo,k1_hi, k2_lo, k2_hi, k3_shared
initialParameters = np.array([100.0, -100.0, 100.0, -100.0, 100.0])

# curve fit the combined data to the combined function
fittedParameters, pcov = curve_fit(combinedFunction, allX, allY, initialParameters)

# values for display of fitted function
k1_lo,k1_hi, k2_lo, k2_hi, k3_shared = fittedParameters

y_fit_lo = funcLow(xLow, k1_lo, k2_lo, k3_shared) # low data set, low equation
y_fit_hi = funcHigh(xHigh, k1_hi, k2_hi, k3_shared) # hi data set, hi equation

plt.plot(allX, allY) # plot the raw data
plt.plot(xLow, y_fit_lo) # plot the low equation using the fitted parameters
plt.plot(xHigh, y_fit_hi) # plot the high equation using the fitted parameters
plt.show()

print('k1_lo, k2_lo:', k1_lo, k2_lo)
print('k1_hi, k2_hi:', k1_hi, k2_hi)
print('k3_shared:', k3_shared)
