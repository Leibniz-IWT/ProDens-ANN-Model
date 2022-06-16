import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from rescale import Rescale_Density
from configuration import filename_forward_model

input_checkpoint = int(input('Enter checkpoint Iteration: '))*1000
filename_forward_model = 'checkpoint_ep' + str(input_checkpoint) + '.pth'

##############
# load model #
##############

# if os.path.isfile('model_forward.pt'):
#     model = torch.load('model_forward.pt')

if os.path.isfile(filename_forward_model):
    checkpoint = torch.load(filename_forward_model)
    from model_invert import model_forward
    model_forward.load_state_dict(checkpoint['model_state_dict'])


###################################
# load data loaders from training #
###################################

if os.path.isfile('dataloader_train.pt'):
    dataLoader_train = torch.load('dataloader_train.pt')
else:
    print('No dataloader for Training found')

if os.path.isfile('dataloader_test.pt'):
    dataloader_test = torch.load('dataloader_test.pt')
else:
    print('No dataloader for testing found')


#####################
# predict densities #
#####################

unscaled_test = []
unscaled_test_target = []

for i, (input_test, target_test) in enumerate(dataloader_test):
    out_test = model_forward(input_test)
    unscaled_out_test = Rescale_Density(out_test)
    unscaled_test.append(unscaled_out_test)

    unscaled_target = Rescale_Density(target_test)
    unscaled_test_target.append(unscaled_target)


unscaled_train = []
unscaled_train_target = []

for j, (input_train, target_train) in enumerate(dataLoader_train):
    unscaled_out_train = Rescale_Density(model_forward(input_train))
    unscaled_train.append(unscaled_out_train)
    unscaled_train_target.append(Rescale_Density(target_train))


unscaled_test = np.concatenate(unscaled_test)
unscaled_test_target = np.concatenate(unscaled_test_target)

unscaled_train = np.concatenate(unscaled_train)
unscaled_train_target = np.concatenate(unscaled_train_target)

average_test = np.average(unscaled_test)
average_test_target = np.mean(unscaled_test_target)

average_train_target = np.mean(unscaled_train_target)

r2_test = np.sum((unscaled_test - average_test_target)**2)/np.sum((unscaled_test_target - average_test_target)**2)
mae_test = np.mean(np.abs(unscaled_test_target-unscaled_test))
max_error_test = np.max(np.abs(unscaled_test_target-unscaled_test))
mse_test = np.mean((unscaled_test_target-unscaled_test)**2)
pe_test = mae_test/np.mean(unscaled_test_target)*100

r2_train = np.sum((unscaled_train-average_train_target)**2)/np.sum((unscaled_train_target - average_train_target)**2)
mae_train = np.mean(np.abs(unscaled_train_target-unscaled_train))
max_error_train = np.max(np.abs(unscaled_train_target-unscaled_train))
mse_train = np.mean((unscaled_train_target-unscaled_train)**2)
pe_train = mae_train/np.mean(unscaled_train_target)*100

print('Model: ', filename_forward_model, '\n Test -- PE: {:.2f}%, MAE: {:.2f}, R2: {:.2f}%, MSE: {:.2f}, Max Error: {:.2f}'.format(pe_test, mae_test, r2_test, mse_test, max_error_test), '\nTrain -- PE: {:.2f}%, MAE: {:.2f}, R2: {:.2f}%, MSE: {:.2f}, Max Error: {:.2f}'.format(pe_train, mae_train, r2_train, mse_train, max_error_train))



######################################
# plotting training and test results #
######################################

plt.plot(unscaled_train_target, unscaled_train, 'ro')
plt.plot(unscaled_test_target, unscaled_test, 'go')
plt.axis([98, 100, 98, 100])
plt.ylabel('predicted')
plt.xlabel('target')
plt.show()

np.savetxt("predicted_test.csv", unscaled_test, delimiter=",")
np.savetxt("target_test.csv", unscaled_test_target, delimiter=",")

np.savetxt("predicted_train.csv", unscaled_train, delimiter=",")
np.savetxt("target_train.csv", unscaled_train_target, delimiter=",")
