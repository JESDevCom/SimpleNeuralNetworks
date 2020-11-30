"""
Modification Author: John Schulz
Original-Author: Dr. Mohammad Imtiaz, Bradley University, ECE
Date Created: 11/19/2020

Description:
    Neural Network Architecture:
        2 input nodes, 1 hidden layer w/ 3 nodes, 1 output node
    Activation Function:
        Sigmoid
    Cost Function:
        
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from MSI_batchNN import MSI_batchNN

tic = time.time()  # Start timer

# ========================================================================
LearningRate = 0.1  # Skips made by gradient adjustment
noEpochs = 1000     # Number of iterations of training
plotRate = 10      # How often to test and display the network performance
# =========================================================================

# Test Data Set - Self Explantory
inputDataAND = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])

inputDataOR = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

inputDataXOR = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

# Pick one data set for this network
inputData = inputDataOR

# =========================================================================
# Weights
#       Assign weights randomly at start but not zeros! why?
#       A baseline of values for wights must be made
w1 = 2*np.random.rand(3, 3)-1  # Weights layer 0 to 1
w2 = 2*np.random.rand(1, 4)-1  # Weights layer 1 to 2

# ========================================================================
# Start training
saveData = np.empty((0, 2))  # Setup matrix to save training data
k = 0
for n in range(0, noEpochs):
    # Run one for each epoch and update the weights
    w1, w2, J, h, y, delta1, delta2 = MSI_batchNN(inputData, w1, w2, LearningRate)

    # test the network occasionally and save data
    if np.mod(n, plotRate) == 0:
        saveData = np.vstack((saveData, np.array([k, J])))
        k = k + 1
        print('Epoch no:' + str(n) + ', J=' + str(J))
        [w1, w2, J, h, y, delta1, delta2] = MSI_batchNN(np.array([[0, 0, 0]]), w1, w2, LearningRate)
        print('[0 0] -> ' + str(h))
        [w1, w2, J, h, y, delta1, delta2] = MSI_batchNN(np.array([[0, 1, 0]]), w1, w2, LearningRate)
        print('[0 1] -> ' + str(h))
        [w1, w2, J, h, y, delta1, delta2] = MSI_batchNN(np.array([[1, 0, 0]]), w1, w2, LearningRate)
        print('[1 0] -> ' + str(h))
        [w1, w2, J, h, y, delta1, delta2] = MSI_batchNN(np.array([[1, 1, 1]]), w1, w2, LearningRate)
        print('[1 1] -> ' + str(h))

# ==========================================================================
# End Time Measurement
toc = time.time()
print('\n\nElapse Time: %f [s]' % (toc-tic))

# ==========================================================================
# Plot [J] the Cost Function
plt.figure(0)
plt.scatter(saveData[:, 0], saveData[:, 1])
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Calculated Cost')
plt.xlim([0, np.size(saveData, 0)])
plt.grid()
plt.show()





