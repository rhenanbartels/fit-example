import numpy as np


x = np.loadtxt('x.txt')
y = np.loadtxt('y.txt')


# best fit to this data was
# y = a - (b*X) - arctan(c/(X-d))
# see fit_derivative.png  and
# fit_deriv_error.png
for i in range(len(x)):
    if i == 0:
        x1 = x[i]
        y1 = y[i]
        continue
    else:
        x2 = x[i]
        y2 = y[i]
        
        dydx = (y2-y1) / (x2-x1)
        
        # you could send this to a file
        print(x[i], dydx)
        
        x1 = x[i]
        y1 = y[i]
