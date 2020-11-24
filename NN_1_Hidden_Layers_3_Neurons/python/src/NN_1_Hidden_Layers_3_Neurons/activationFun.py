"""
Author: John Schulz
Co-Author: Dr. Mohammad Imtiaz, Bradley University, ECE
Date Created: 11/19/2020

Description:
    Activation Function is sigmoid.
"""

import numpy as np

def activationFun(x):

    # Sigmoid
    return 1/(1 + np.exp(x))