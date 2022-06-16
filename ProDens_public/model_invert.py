import torch
import torch.nn as nn
import os
import torch.nn.functional as F


###############
# setup model #
###############

class Model_Backward (nn.Module):
    def __init__(self):
        super(Model_Backward, self).__init__()
        self.lin1 = nn.Linear(2, 40)
        self.lin2 = nn.Linear(40, 60)
        self.lin3 = nn.Linear(60, 3)
        #self.dropout = nn.Dropout(p=0.15)


    def forward(self, x):
        x = torch.relu(self.lin1(x))
        #x = self.dropout(x)
        x = torch.relu(self.lin2(x))
        #x = self.dropout(x)
        x = torch.sigmoid(self.lin3(x))
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num


class Model_Forward (nn.Module):
    def __init__(self):
        super(Model_Forward, self).__init__()
        self.linf1 = nn.Linear(4, 60)
        self.linf2 = nn.Linear(60, 40)
        self.linf3 = nn.Linear(40, 30)
        self.linf4 = nn.Linear(30, 1)
        #self.linf4 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.linf1(x))
        x = torch.relu(self.linf2(x))
        x = torch.relu(self.linf3(x))
        x = torch.sigmoid(self.linf4(x))
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num


model_backward = Model_Backward()
model_forward = Model_Forward()


##############
# load model #
##############

if os.path.isfile('model_forward.pth'):
    checkpoint = torch.load('model_forward.pth')
    from model_invert import model_forward
    model_forward.load_state_dict(checkpoint['model_state_dict'])
