"""
Author: John Schulz
Co-Author: Dr. Mohammad Imtiaz, Bradley University, ECE
Date Created: 11/19/2020

Description:
    Cost Function: Mean Square Error (MSE)
"""

import numpy as np


def costFun(y, h):
    J = 0
    h = h[0, 0]

    # Reduce Undefined Problem of Log(0) = NaN
    if h == 1:
        z = 0.99999999999999999999
        J = ((y * np.log(h)) + ((1 - y) * np.log(1 - z)))
    elif h == 0:
        z = 0.00000000000000000001
        J = ((y * np.log(z)) + ((1 - y) * np.log(1 - h)))
    else:
        J = ((y * np.log(h)) + ((1 - y) * np.log(1 - h)))

    return J