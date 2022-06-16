import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import pandas as pd

from model_invert import model_forward, model_backward


###################
# general Setting #
###################

epochs = 1000
iterations = 200
alpha = 0.001
beta = 0.2 # influence of build rate hint for parameter prediction

#L1 = Mean Absolute Error, MSE = Mean Squared Error
criterion = nn.L1Loss()
criterion_backward = nn.L1Loss()
# Adam or SGD
optimizer_forward = optim.Adam(model_forward.parameters(), lr=alpha)
optimizer_backward = optim.Adam(model_backward.parameters(), lr=alpha)


###########################################################
# Save epochs applied and create name for checkpoint save #
###########################################################

if os.path.isfile('epoch.csv'):
    epochs_applied = int(pd.read_csv('epoch.csv', delimiter=',', header=None)[0][0])
    print('Applied Epochs: ', epochs_applied)
else:
    epochs_applied = 0
    np.savetxt("epoch.csv", [epochs_applied], delimiter=",")

# filename_forward_model_new = 'checkpoint_ep'+str(epochs_applied+epochs)+'.pth'
filename_forward_model = 'checkpoint_ep'+str(epochs_applied)+'.pth'