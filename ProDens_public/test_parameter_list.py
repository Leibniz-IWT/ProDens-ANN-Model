import os
import torch
import tqdm
import numpy as np
from calculate_dataspace import Find_Density_Test, Get_Parameterspace, Find_Density
from rescale import *
import matplotlib.pyplot as plt

# load dataloader_test
if os.path.isfile('dataloader_test.pt'):
    dataloader_test = torch.load('dataloader_test.pt')
else:
    print('No dataloader for testing found')


densities_res = []
densities_exp = []
parameters_exp = []
br_res = []; sd_res = []; ll_res = []; ss_res = []; hd_res = []

for i, (input, target) in enumerate(dataloader_test):
    target = Rescale_Density(target)[0][0]
    dataspace = Get_Parameterspace()
    result = Find_Density_Test(target, dataspace)
    densities_res.append(result[4])
    br_res.append(result[5])
    sd_res.append(result[0]); ll_res.append(result[1]); ss_res.append(result[2]); hd_res.append(result[3])

    densities_exp.append(target)
    parameters_exp.append(input.detach().numpy())

    print('Parameter Set {}/{} [{:.2f}%, {:.2f}mm] calculated!'.format(i+1, len(dataloader_test.dataset), target, result[0]), end='\r')


parameters_exp = np.concatenate(parameters_exp)
sd_exp = Rescale_Thickness(parameters_exp[:, 0])
ll_exp = Rescale_Laser(parameters_exp[:, 1])
ss_exp = Rescale_Speed(parameters_exp[:, 2])
hd_exp = Rescale_Hatch(parameters_exp[:, 3])
br_exp = sd_exp*ss_exp*hd_exp
ved_exp = ll_exp/(ss_exp*hd_exp*sd_exp)


parameters_res = np.asarray(np.stack([sd_res, ll_res, ss_res, hd_res, densities_res], axis=1))
parameters_exp = np.stack([sd_exp, ll_exp, ss_exp, hd_exp, densities_exp], axis=1)

ved_res = parameters_res[:, 1]/(parameters_res[:, 0]*parameters_res[:, 2]*parameters_res[:, 3])

np.savetxt('results.csv', parameters_res, delimiter=';')
np.savetxt('experiment.csv', parameters_exp, delimiter=';')


####################
# Plotting Results #
####################

fig, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]) = plt.subplots(2, 4)

ax1.plot(densities_exp, densities_res, 'go')
#ax1.plot(unscaled_target_rand, unscaled_densities_rand, 'bo')
ax1.axis([70, 100, 70, 100])
ax1.set(xlabel="traget", ylabel='predicted')

ax2.plot(densities_exp, br_exp, 'ro')
ax2.plot(densities_exp, br_res, 'go')
ax2.axis([70, 100, 0, 80])
ax2.set(xlabel='density', ylabel='build rate')

ax3.plot(densities_exp, ved_exp, 'ro')
ax3.plot(densities_exp, ved_res, 'go')
ax3.axis([70, 100, 0, 60])
ax3.set(xlabel='density', ylabel='volume energy density')

ax5.plot(ll_exp, densities_exp, 'ro')
ax5.plot(ll_res,  densities_exp, 'go')
ax5.axis([100, 400, 70, 100])
ax5.set(xlabel='Laser Power', ylabel='target density')

ax6.plot(hd_exp,  densities_exp, 'ro')
ax6.plot(hd_res,  densities_exp, 'go')
ax6.axis([0, 0.3, 70, 100])
ax6.set(xlabel='Hatch Distance', ylabel='target density')

ax7.plot(ss_exp,  densities_exp, 'ro')
ax7.plot(ss_res,  densities_exp, 'go')
# ax7.axis([300, 2600, 70, 100])
ax7.set(xlabel='Scan Speed', ylabel='target density')

ax8.plot(sd_exp,  densities_exp, 'ro')
ax8.plot(sd_res,  densities_exp, 'go')
ax8.axis([0, 0.1, 70, 100])
ax8.set(xlabel='layer thickness', ylabel='target density')


plt.show()

