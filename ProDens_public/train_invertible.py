import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim


from datapreparation import dataloader_train, dataloader_test
from model_invertible import model_inn
from model_invertible import MMDLoss


################################################
# Configuration of the INN Training Parameters #
################################################

epochs = 100
lr = 0.001
optimizer = 'Adam'
alpha = 0.5
beta = 0.5

loss_y = nn.L1Loss()
loss_z = MMDLoss()


if optimizer=='Adam':
    optimizer_inn = optim.Adam(model_inn.parameters(), lr=lr)
else:
    if optimizer == 'SGD':
    optimizer_inn = optim.SGD(model_inn.parameters(), lr=lr)


