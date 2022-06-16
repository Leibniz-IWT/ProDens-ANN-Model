import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from rescale import *
from configuration import optimizer_forward, optimizer_backward
import pandas as pd
from datapreparation import min_SD, max_SD, min_D, max_D, min_LL, max_LL, min_SS, max_SS, min_HD, max_HD, low, high


input_checkpoint = int(input('Enter checkpoint Iteration: '))*1000
filename_forward_model = 'checkpoint_ep' + str(input_checkpoint) + '.pth'


##############
# load model #
##############

if os.path.isfile('model_forward.pth'):
    checkpoint = torch.load('model_forward.pth')
    from model_invert import model_forward
    model_forward.load_state_dict(checkpoint['model_state_dict'])
    optimizer_forward.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    print('No forward model found!')
    sys.exit()

if os.path.isfile(filename_forward_model):
    checkpoint_back = torch.load(filename_forward_model)
    from model_invert import model_backward
    model_backward.load_state_dict(checkpoint_back['model_state_dict'])
    optimizer_backward.load_state_dict(checkpoint_back['optimizer_state_dict'])
else:
    print('No Parameter Prediction Model {} found!'.format(filename_forward_model))
    sys.exit()

model_backward.eval()


####################
# create test data #
####################

density = float(input('Enter lower border for density: '))
input_max_dens = float(input('Enter upper border for density: '))
input_step_size = float(input('Enter step size for densities: '))
input_layer_thickness = float(input('Enter layer thickness: '))

density_list = []
layer_thickness = []

while density <= input_max_dens:
    density_list.append(density)
    layer_thickness.append(input_layer_thickness)
    density = density + input_step_size

def normalize(x, min, max):
    out = []
    for element in x:
        val = (element-min)/(max-min) * (high-low) + low
        out.append(val)
    return out

density_list = normalize(density_list, min_D, max_D)
layer_thickness = normalize(layer_thickness, min_SD, max_SD)


density = torch.Tensor(density_list).to(torch.float).view(len(density_list), 1)
thickness = torch.Tensor(layer_thickness).to(torch.float).view(len(layer_thickness), 1)

test_data = torch.cat((thickness, density), 1)


################################
# testing model with test data #
################################

# prediction of test parameters and densities
unscaled_density_test_out = []
unscaled_density_test_target = []
parameters_test_out_l = []
parameters_test_target_l = []

for i in enumerate(test_data):
    X = test_data

    out = model_backward(X)

    # density_backward = model_forward(model_backward(label_back))
    y1 = torch.Tensor(out[:, 0]).to(torch.float).view(len(out[:, 0]), 1)
    y2 = torch.Tensor(out[:, 1]).to(torch.float).view(len(out[:, 1]), 1)
    y3 = torch.Tensor(out[:, 2]).to(torch.float).view(len(out[:, 2]), 1)

    Y = torch.cat((thickness, y1, y2, y3), 1)

    SD = Y[:, 0]
    SS = Y[:, 2]
    HA = Y[:, 3]

    parameters_test_out = Y

    density_test_out = model_forward(Y)

    unscaled_density_test_out.append(Rescale_Density(density_test_out))
    unscaled_density_test_target.append(Rescale_Density(density))

    parameters_test_out_l.append(parameters_test_out.detach().numpy())

unscaled_density_test_out = np.concatenate(unscaled_density_test_out)[:, 0]
unscaled_density_test_target = np.concatenate(unscaled_density_test_target)[:, 0]

parameters_test_out_l = np.concatenate(parameters_test_out_l)

SD_pred = Rescale_Thickness(parameters_test_out_l[:, 0])
LL_pred = Rescale_Laser(parameters_test_out_l[:, 1])
SS_pred = Rescale_Speed(parameters_test_out_l[:, 2])
HA_pred = Rescale_Hatch(parameters_test_out_l[:, 3])

BR_pred = SD_pred * SS_pred * HA_pred
VED_pred = LL_pred / (SD_pred * SS_pred * HA_pred)
SED_pred = LL_pred / SS_pred

mean_BR = np.mean(BR_pred)

print('mean build rate: {:.2f}'.format(mean_BR))


# save data table

parameters_res = np.asarray(np.stack([SD_pred, LL_pred, SS_pred, HA_pred, unscaled_density_test_out, unscaled_density_test_target], axis=1)) #hier noch Spalte mit Zielwerten einfügen

np.savetxt('results_parameter_prediction_test.csv', parameters_res, delimiter=';')


#########################################
# plotting build rate and volume energy #
#########################################

fig, ([ax1, ax2]) = plt.subplots(1, 2)

ax1.plot(BR_pred, unscaled_density_test_out, 'go')
ax1.set(xlabel="build rate", ylabel='predicted density')

ax2.plot(VED_pred, unscaled_density_test_out, 'ro')
ax2.set(xlabel='volume energy', ylabel='predicted density')

plt.tight_layout()
plt.show()