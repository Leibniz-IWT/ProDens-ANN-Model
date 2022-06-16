import numpy as np
import sys
import os

import pandas as pd
import torch
from configuration import optimizer_forward, optimizer_backward
from rescale import *


if os.path.isfile('model_forward.pth'):
    checkpoint = torch.load('model_forward.pth')
    from model_invert import model_forward
    model_forward.load_state_dict(checkpoint['model_state_dict'])
    optimizer_forward.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    print('No forward model found!')
    sys.exit()


def Create_Dataspace(steps, low, high):
    counter = low
    stepsize = (high-low)/steps
    sd = []
    ll = []
    ss = []
    hd = []
    database=[]

    while counter < high:
        sd.append(counter)
        ll.append(counter)
        ss.append(counter)
        hd.append(counter)
        counter += stepsize

    print('\n\nCombine Vectors for Database creation.', end='\r')

    for idx1, a in enumerate(sd):
        for idx2, b in enumerate(ll):
            for idx3, c in enumerate(ss):
                for idx4, d in enumerate(hd):
                    database.append([a, b, c, d])
            print('Combine Vectors for Database creation {:.0f}/{:.0f}'.format(len(database), steps**4), end='\r')

    return database # [[a, b, c, d] for a in sd for b in ll for c in ss for d in hd]


def Get_Density(features):
    features = torch.Tensor(features)
    densities = []

    print('\n\nCalculate Densities for Database.', end='\r')

    for idx, feature in enumerate(features):
        print('Calculate Densities for Database {}/{}'.format(idx, len(features)), end='\r')
        density = model_forward(feature)
        densities.append(Rescale_Density(density))

    return densities


def Find_Density(value, thickness, array):
    low_density = value - 0.05
    high_density = value + .2
    low_thickness = thickness - .005
    high_thickness = thickness + .005

    array_t = array[:, 0]
    sd = []; ll =[]; ss = []; hd = []; d = []; br =[]


    for idx_thickness in range(len(array)):
        if low_thickness <= array_t[idx_thickness] <= high_thickness:
            sd.append(array[idx_thickness, 0])
            ll.append(array[idx_thickness, 1])
            ss.append(array[idx_thickness, 2])
            hd.append(array[idx_thickness, 3])
            d.append(array[idx_thickness, 4])
            br.append(array[idx_thickness, 5])

    if len(sd)==0:
        print('\nNo Layer Thickness of {:.2f} mm found'.format(thickness))
        sys.exit()

    parameters_sorted = np.stack([sd, ll, ss, hd, d, br], axis=1)
    sd.clear(), ll.clear(), ss.clear(), hd.clear(), d.clear(), br.clear()

    array_s = parameters_sorted[:, 4]

    for idx_density in range(len(parameters_sorted)):
        if low_density <= array_s[idx_density] <= high_density:
            sd.append(parameters_sorted[idx_density, 0])
            ll.append(parameters_sorted[idx_density, 1])
            ss.append(parameters_sorted[idx_density, 2])
            hd.append(parameters_sorted[idx_density, 3])
            d.append(parameters_sorted[idx_density, 4])
            br.append(parameters_sorted[idx_density, 5])

    if len(sd)==0:
        print('\nNo Density of {:.2f} % found for Layer Thickness of {:.2f} mm'.format(value, thickness))
        sys.exit()

    parameters_output = np.stack([sd, ll, ss, hd, d, br], axis=1)
    idx_best = np.argmax(parameters_output[:, 5])
    print('\n{} combinations checked.'.format(len(array)))
    print('{} possible Solutions for Your Inquest.'.format(len(parameters_output)))

    return parameters_output[idx_best]


def Find_Density_Test(value, array):
    low_density = value - 0.1
    high_density = value + .2

    array_s = array[:, 4]
    sd = []; ll =[]; ss = []; hd = []; d = []; br =[]

    for idx_density in range(len(array)):
        if low_density <= array_s[idx_density] <= high_density:
            sd.append(array[idx_density, 0])
            ll.append(array[idx_density, 1])
            ss.append(array[idx_density, 2])
            hd.append(array[idx_density, 3])
            d.append(array[idx_density, 4])
            br.append(array[idx_density, 5])

    if len(sd)==0:
        print('\nNo Density of {:.2f} % found'.format(value))
        sys.exit()

    parameters_output = np.stack([sd, ll, ss, hd, d, br], axis=1)
    idx_best = np.argmax(parameters_output[:, 5])

    return parameters_output[idx_best]


def Get_Parameterspace():
    if os.path.isfile('Database_Density_Prediction-Mika_Altmann.csv'):
        dataspace = pd.read_csv('Database_Density_Prediction-Mika_Altmann.csv', delimiter=';', header=None)
        sd = np.array(dataspace[0])
        ll = np.array(dataspace[1])
        ss = np.array(dataspace[2])
        hd = np.array(dataspace[3])
        d = np.array(dataspace[4])
        br = np.array(dataspace[5])

        dataspace = np.stack([sd, ll, ss, hd, d, br], axis=1)

    else:
        steps = float(input('Number of Steps: '))
        dataspace = np.array(Create_Dataspace(steps, 0, 1))
        densities = Get_Density(dataspace)

        sd = Rescale_Thickness(dataspace[:, 0])
        ll = Rescale_Laser(dataspace[:, 1])
        ss = Rescale_Speed(dataspace[:, 2])
        hd = Rescale_Hatch(dataspace[:, 3])
        d = np.concatenate(densities)
        br = sd*ss*hd

        dataspace = np.stack([sd, ll, ss, hd, d, br], axis=1)
        np.savetxt('Database_Density_Prediction-Mika_Altmann.csv', dataspace, delimiter=';')

    return dataspace


