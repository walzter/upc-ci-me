# net
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.transforms as transforms
# splitting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# nice load bar
from tqdm import tqdm
#linalg + df
import pandas as pd
import numpy as np
# viz
import matplotlib.pyplot as plt
# nice color
import termcolor
from termcolor import colored

# from the utils
from cnn_utils import load_data_split
from CNN_model import ConvNet

# TRAINING
from train_feedback import train_check_loop
# TESTING
from test_feedback import test_check_loop
# VALIDATION
from val_feedback import val_check_loop

CONFIG_DICT = {
			    'PATH_TO_FILE': './all_labels.csv',
			    'TRAIN_RATIO': 0.7,
			    'VALIDATION_RATIO': 0.8,
			    'BATCH_SIZE': 16,
			    'TO_SHUFFLE': True,
			    'EPOCHS': 100,
			    'N_BATCHES': 20,
			    'LEARNING_RATE': 0.1,
			    'MOMENTUM': 0.9,
			    'WEIGHT_DECAY':0.8
    			}


def main(CONFIG_DICT):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # read the file path
    FILE_PATH = ''
    # splitting the data
    (train_dataset, test_dataset, val_dataset), (train_loader, test_loader, val_loader) = load_data_split(CONFIG_DICT['PATH_TO_FILE'],
                                                                                                          CONFIG_DICT['TRAIN_RATIO'],
                                                                                                          CONFIG_DICT['VALIDATION_RATIO'],
                                                                                                          CONFIG_DICT['BATCH_SIZE'],
                                                                                                          CONFIG_DICT['TO_SHUFFLE'])

    # instantiate the ConvNet
    net = ConvNet()
    if device.type == 'cuda':
        net = net.cuda()
    else:
        net = net.to(device)
    # define the optimizer
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=CONFIG_DICT['LEARNING_RATE'],
                                momentum=CONFIG_DICT['MOMENTUM'])
    opt = Adam(net.parameters(), lr=CONFIG_DICT['LEARNING_RATE'])
    # define the criteria
    criteria = nn.CrossEntropyLoss()

    for i in range(1, CONFIG_DICT['EPOCHS']+1):
        train_check_loop(train_loader, device, net, opt, criteria, i)
        test_check_loop(test_loader, device, net, opt,criteria, i, CONFIG_DICT['N_BATCHES'])
        val_check_loop(val_loader, device, net, opt,criteria, i, CONFIG_DICT['N_BATCHES'])


if __name__ == "__main__":
    main(CONFIG_DICT)
