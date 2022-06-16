import os
import torch
import numpy as np
import pandas as pd
from rescale import Rescale_Density

data = pd.read_csv('data.csv', delimiter=";")
length = len(data)

sd = np.reshape(np.array(data['Schichtdicke']), (1, -1))
ll = np.reshape(np.array(data['Laserleistung']), (1, -1))
ss = np.reshape(np.array(data['Scangeschwindigkeit']), (1, -1))
hd = np.reshape(np.array(data['Hatchabstand']), (1, -1))

min_SD = 0.025; max_SD = 0.1
min_LL = 152; max_LL = 350
min_SS = 803; max_SS = 1600
min_HD = 0.07; max_HD = 0.15
min_D = 98; max_D = 100

low = 0; high = 1

normalized_SD = (sd-min_SD)/(max_SD-min_SD)*(high-low)+low
normalized_LL = (ll-min_LL)/(max_LL-min_LL)*(high-low)+low
normalized_SS = (ss-min_SS)/(max_SS-min_SS)*(high-low)+low
normalized_HD = (hd-min_HD)/(max_HD-min_HD)*(high-low)+low

sd = torch.Tensor(normalized_SD).to(torch.float).view(length, 1)
ll = torch.Tensor(normalized_LL).to(torch.float).view(length, 1)
ss = torch.Tensor(normalized_SS).to(torch.float).view(length, 1)
hd = torch.Tensor(normalized_HD).to(torch.float).view(length, 1)

x = torch.cat((sd, ll, ss, hd), 1)

if os.path.isfile('model_forward.pth'):
    checkpoint = torch.load('model_forward.pth')
    from model_invert import model_forward
    model_forward.load_state_dict(checkpoint['model_state_dict'])

model = model_forward

density = model(x)

density = Rescale_Density(density)

np.savetxt("Ti64-Densities.csv", density, delimiter=";")



