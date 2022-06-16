import numpy
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import os


#######################################
#        LOAD EXPERIMENTAL DATA       #
#                                     #
# - Training Data is already splitted #
#                                     #
#######################################

data_Train = pd.read_csv('Ti64_train_split.csv', delimiter=';')
data_Test = pd.read_csv('Ti64_test_split.csv', delimiter=';')

training_length = len(data_Train)
test_length = len(data_Test)


#####################################
# normalize all needed inputs (0,1) #
#####################################

sd_Train = np.reshape(np.array(data_Train['Schichtdicke']), (1, -1))
ll_Train = np.reshape(np.array(data_Train['Laserleistung']), (1, -1))
ss_Train = np.reshape(np.array(data_Train['Scangeschwindigkeit']), (1, -1))
hd_Train = np.reshape(np.array(data_Train['Hatchabstand']), (1, -1))
d_Train = np.reshape(np.array(data_Train['Dichte']), (1, -1))

sd_Test = np.reshape(np.array(data_Test['Schichtdicke']), (1, -1))
ll_Test = np.reshape(np.array(data_Test['Laserleistung']), (1, -1))
ss_Test = np.reshape(np.array(data_Test['Scangeschwindigkeit']), (1, -1))
hd_Test = np.reshape(np.array(data_Test['Hatchabstand']), (1, -1))
d_Test = np.reshape(np.array(data_Test['Dichte']), (1, -1))


### Definition von Minima, Maxima und Intervallgrenzen für die Skalierung
min_SD = 0.01; max_SD = 0.11
min_LL = 100; max_LL = 400
min_SS = 700; max_SS = 1700
min_HD = 0.05; max_HD = 0.2
min_D = 97; max_D = 100

low = -1; high = 1

normalized_SD_Train = (sd_Train-min_SD)/(max_SD-min_SD)*(high-low)+low
normalized_LL_Train = (ll_Train-min_LL)/(max_LL-min_LL)*(high-low)+low
normalized_SS_Train = (ss_Train-min_SS)/(max_SS-min_SS)*(high-low)+low
normalized_HD_Train = (hd_Train-min_HD)/(max_HD-min_HD)*(high-low)+low
normalized_D_Train = (d_Train-min_D)/(max_D-min_D)*(high-low)+low

normalized_SD_Test = (sd_Test-min_SD)/(max_SD-min_SD)*(high-low)+low
normalized_LL_Test = (ll_Test-min_LL)/(max_LL-min_LL)*(high-low)+low
normalized_SS_Test = (ss_Test-min_SS)/(max_SS-min_SS)*(high-low)+low
normalized_HD_Test = (hd_Test-min_HD)/(max_HD-min_HD)*(high-low)+low
normalized_D_Test = (d_Test-min_D)/(max_D-min_D)*(high-low)+low


#####################################################
# create tensor of features and target for ML Model #
#####################################################

sd_Train = torch.Tensor(normalized_SD_Train).to(torch.float).view(training_length, 1)
ll_Train = torch.Tensor(normalized_LL_Train).to(torch.float).view(training_length, 1)
ss_Train = torch.Tensor(normalized_SS_Train).to(torch.float).view(training_length, 1)
hd_Train = torch.Tensor(normalized_HD_Train).to(torch.float).view(training_length, 1)
d_Train = torch.Tensor(normalized_D_Train).to(torch.float).view(training_length, 1)

sd_Test = torch.Tensor(normalized_SD_Test).to(torch.float).view(test_length, 1)
ll_Test = torch.Tensor(normalized_LL_Test).to(torch.float).view(test_length, 1)
ss_Test = torch.Tensor(normalized_SS_Test).to(torch.float).view(test_length, 1)
hd_Test = torch.Tensor(normalized_HD_Test).to(torch.float).view(test_length, 1)
d_Test = torch.Tensor(normalized_D_Test).to(torch.float).view(test_length, 1)

x_Train = torch.cat((sd_Train, ll_Train, ss_Train, hd_Train), 1)
y_Train = d_Train

x_Test = torch.cat((sd_Test, ll_Test, ss_Test, hd_Test), 1)
y_Test = d_Test


##################
# create Dataset #
##################

class Densities_Train (Dataset):

    def __init__(self):
        self.x = x_Train
        self.y = y_Train
        self.n_samples = x_Train.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class Densities_Test (Dataset):

    def __init__(self):
        self.x = x_Test
        self.y = y_Test
        self.n_samples = x_Test.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset_Train = Densities_Train()
dataset_Test = Densities_Test()


#################################
#         split dataset         #
#################################

#dataset_Train, dataset_Test = torch.utils.data.random_split(dataset, [training_length, test_length])


#####################
# create dataloader #
#####################

dataloader_train = DataLoader(dataset=dataset_Train, batch_size=40, shuffle=True)
dataloader_test = DataLoader(dataset=dataset_Test)


# ####################
# # load data loader #
# ####################
#
# if os.path.isfile('dataloader_train.pt'):
#     dataloader_train = torch.load('dataloader_train.pt')
#
# if os.path.isfile('dataloader_test.pt'):
#     dataloader_test = torch.load('dataloader_test.pt')


###############################################
# rescale density output to experimental room #
###############################################

def Rescale(x):
    x = x.detach().numpy() * D.max()
    return x

