import os
import pandas as pd
from torchvision.io import read_image
import cv2
from torch.utils.data import Dataset

class CovidCoughDatasetSpectrograms(Dataset):
    def __init__(self,label, data):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (32,32),interpolation=cv2.INTER_AREA)
        image = (image/255.)
        label = self.label[idx]
        return image, label