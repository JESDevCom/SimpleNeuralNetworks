import numpy as np
from costFun import costFun
from forwardProp import forwardProp



def MSI_batchNN(inputData, w1, w2, LearningRate):

    temp_delta1 = 0
    temp_delta2 = 0
    delta1 = 0
    delta2 = 0
    h = 0
    y = 0
    J = 0
    n = 0

    for n in range(0, np.size(inputData, 0)):

        # take row 1 of the input data and make it a column vector
        a0 = np.transpose(np.concatenate(([1], inputData[n, 0:2]))[np.newaxis])

        # Forward Propagation
        a1 = forwardProp(w1, a0)
        a2 = forwardProp(w2, np.concatenate((np.array([[1]]), a1)))
        h = a2

        # Back Propagation
        # Calculate Error at the output layer
        y = inputData[n, 2]

        # error in layer 2
        delta2 = h - y

        # error in layer 1
        delta1t = np.multiply( np.transpose(w2) * delta2, np.multiply(np.concatenate((np.array([[1]]), a1)), 1-np.concatenate((np.array([[1]]), a1)) ))
        delta1  = delta1t[1:np.size(delta1t, 0)]

        # Accumulate partial derivative
        temp_delta2 = temp_delta2 + (delta2 * np.transpose(np.concatenate((np.array([[1]]), a1))))
        temp_delta1 = temp_delta1 + (delta1 * np.transpose(a0))

        J = J + costFun(y, h)

    n = n + 1

    J = J/(-1*n)

    #Adjust Weights
    w2_new = w2 - (LearningRate * (temp_delta2 / n))  # w1_new
    w1_new = w1 - (LearningRate * (temp_delta1 / n))  # w2_new

    return w1_new, w2_new, J, h, y, delta1, delta2