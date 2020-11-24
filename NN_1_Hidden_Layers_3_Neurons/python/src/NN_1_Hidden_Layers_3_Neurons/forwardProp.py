"""
Author: John Schulz
Co-Author: Dr. Mohammad Imtiaz, Bradley University, ECE
Date Created: 11/19/2020

Description:
    Forward propagation function: A*B
"""

from activationFun import activationFun
import numpy as np

def forwardProp(Wc, x):
    z = np.matmul(Wc, x)
    a = activationFun(-1*z)
    return a