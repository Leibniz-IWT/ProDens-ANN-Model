import os.path

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from model_invert import model_backward
from configuration import epochs, criterion_backward, optimizer_backward, beta, iterations, filename_forward_model, optimizer_forward


##################################################
# create hint function for maximizing build rate #
##################################################

### squared hint
def MSQ_BR (x):
    return torch.sum(torch.square(x)-2*x+1) / x.size()[0]

### linear hint
def MLIN_BR (x):
    return torch.sum(x-2*x+1) / x.size()[0]


####################
# open dataloaders #
####################

if os.path.isfile('dataloader_test.pt'):
    dataloader_test = torch.load('dataloader_test.pt')
else:
    from datapreparation import dataloader_test
    torch.save(dataloader_test, 'dataloader_test.pt')

if os.path.isfile('dataloader_train.pt'):
    dataloader_train = torch.load('dataloader_train.pt')
else:
    from datapreparation import dataloader_train


######################
# load forward model #
######################

if os.path.isfile('model_forward.pth'):
    checkpoint = torch.load('model_forward.pth')
    from model_invert import model_forward
    model_forward.load_state_dict(checkpoint['model_state_dict'])
    optimizer_forward.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    print('No forward model found!')

# input_checkpoint = int(input('Enter checkpoint Iteration: '))*1000
# filename_forward_model = 'checkpoint_ep' + str(input_checkpoint) + '.pth'

if os.path.isfile(filename_forward_model):
    checkpoint_back = torch.load(filename_forward_model)
    from model_invert import model_backward
    model_backward.load_state_dict(checkpoint_back['model_state_dict'])
    optimizer_backward.load_state_dict(checkpoint_back['optimizer_state_dict'])
else:
    from model_invert import model_forward
    #print(model_forward)


######################
# main training loop #
######################

for iteration in tqdm(range(iterations)):
    for epoch in range(epochs):
        model_backward.train()
        from datapreparation import dataloader_train

        for j, (input_back, label_back) in enumerate(dataloader_train):

            # resort Data for Backward Training with Density and Thickness
            x1 = torch.Tensor(input_back[:, 0]).to(torch.float).view(len(input_back[:, 0]), 1)
            x2 = torch.Tensor(label_back[:, 0]).to(torch.float).view(len(label_back[:, 0]), 1)
            X = torch.cat((x1, x2), 1)

            out = model_backward(X)

            #density_backward = model_forward(model_backward(label_back))
            y1 = torch.Tensor(out[:, 0]).to(torch.float).view(len(out[:, 0]), 1) # Laserleistung
            y2 = torch.Tensor(out[:, 1]).to(torch.float).view(len(out[:, 1]), 1)
            y3 = torch.Tensor(out[:, 2]).to(torch.float).view(len(out[:, 2]), 1)

            Y = torch.cat((x1, y1, y2, y3), 1)

            density_backward = model_forward(Y)

            SD = X[:, 0]
            SS = Y[:, 1]
            HA = Y[:, 2]

            #out = model_backward(label_back)

            # SD = out[:, 0]
            # SS = out[:, 2]
            # HA = out[:, 3]

            build_rate = SD * SS * HA

            msq_build_rate = beta * MSQ_BR(build_rate) #/ (epoch+1)
            loss_backward = criterion_backward(density_backward, label_back)

            loss_parampred = (loss_backward + msq_build_rate / (iteration*epoch+1)**2) / 2

            model_backward.zero_grad()
            loss_parampred.backward()
            optimizer_backward.step()

    ### Speichern der Modells und des Optimierers, um später weiter trainieren zu können
    filename_forward_model = 'checkpoint_ep' + str((iteration + 1) * epochs) + '.pth'
    torch.save({
        'model_state_dict': model_backward.state_dict(),
        'optimizer_state_dict': optimizer_backward.state_dict(),
    }, filename_forward_model)

    # torch.save(model_forward, 'model_forward.pt')
    torch.save(dataloader_train, 'dataloader_train.pt')


        # print('Train Epoch: {} {:.0f}%, Loss Backward: {:.6f}, Loss Build Rate: {:.6f}'.format(epoch + 1, 100. * (
        #                     epoch + 1) / epochs, loss_backward, msq_build_rate), end='\r')


# print('Train Epoch: {} {:.0f}%, Loss Backward: {:.6f}, Loss Build Rate: {:.6f}'.format(epoch + 1, 100. * (
#                 epoch + 1) / epochs, loss_backward, msq_build_rate))
