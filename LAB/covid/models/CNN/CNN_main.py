# net 
import torch 
import torchvision
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam 
import torchvision.transforms as transforms
#splitting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#nice load bar
from tqdm import tqdm
#linalg + df
import pandas as pd
import numpy as np
#viz
import matplotlib.pyplot as plt

# from the utils 
from cnn_utils import load_data_split
from CNN_model import ConvNet

CONFIG_DICT = {
				'PATH_TO_FILE':,
		  		'TRAIN_RATIO':0.7,
		  		'VALIDATION_RATIO':0.8,
		  		'BATCH_SIZE':16,
		  		'TO_SHUFFLE':True,
		  		'TO_SHUFFLE':,
		  		'TO_SHUFFLE':,
		  		'TO_SHUFFLE':,


		  		}


def main(CONFIG_DICT):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# read the file path 
	FILE_PATH = ''
	# splitting the data 
	(train_dataset, test_dataset, val_dataset),(train_loader, test_loader, val_loader) = load_data_split(CONFIG_DICT['PATH_TO_FILE'],
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




