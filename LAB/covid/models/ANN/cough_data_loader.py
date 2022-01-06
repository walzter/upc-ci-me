import torch
from torch.utils.data import Dataset, DataLoader

class CovidCoughDataset(Dataset):
  
  def __init__(self, data,label, mode='train'):
    self.mode = mode
    self.data = data
    self.label = label
    if self.mode == 'train':
      self.input = torch.from_numpy(self.data)
      self.output = torch.from_numpy(self.label)
    else:
      self.input = self.data
    
  def __len__(self):
    return len(self.input)
  
  def __getitem__(self, idx):
    if self.mode == 'train':
      inpt = self.input[idx]
      oupt = self.output[idx]
      return {'input':inpt,
              'output':oupt}
    else:
      inpt = torch.Tensor(self.input[idx])
      return {'input':inpt}

def gen_data_loaders(data_to_train,labels_to_train,data_to_test,labels_to_test,batch_size, to_shuffle=False):
  '''
  Generates:
    data_train
    data_test

  '''
  # training 
  data_to_train = CovidCoughDataset(data_to_train,labels_to_train)
  data_train = DataLoader(dataset = data_to_train, batch_size = batch_size, shuffle=to_shuffle)
  # testing 
  data_to_test = CovidCoughDataset(data_to_test,labels_to_test)
  data_test = DataLoader(dataset = data_to_test, batch_size = batch_size, shuffle=to_shuffle)
  return data_train, data_test