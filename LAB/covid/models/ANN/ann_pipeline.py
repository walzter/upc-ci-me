
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
# utils
from prepare_data import prep_dataframe, train_test_val_split
from cough_data_loader import CovidCoughDataset, gen_data_loaders
from ann_utils import swish, train_loop, test_loop
from training_testing_loop import _fit_model_train_test_n_batches
from ANN_model import SimpleNetHeLarge


def make_pipeline(CONFIG):

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  print(f"\nUsing the following device: {device}")

  # show the config 
  for k, v in CONFIG.items():
    print(k, v)
  print('\n')

  # getting only the values for the input of the NET
  X_vals, Y_vals = prep_dataframe(CONFIG['PATH'])
  # sepparating the splits
  (X_train, Y_train), (X_test,Y_test)= train_test_val_split(X_vals,Y_vals,TRAIN_RATIO=CONFIG['TRAIN_RATIO'])
  # Preparing the dataloaders for train, test
  data_train, data_test = gen_data_loaders(X_train,Y_train,X_test,Y_test,batch_size=CONFIG['batch_size'],to_shuffle=False)
  # instantiating the NN
  #net = SimpleNet()
  #net = SimpleNetHe()
  net = SimpleNetHeLarge()
  # GPU BABY
  #net = net.to(device)
  net = net.cuda() if device =='cuda' else net
  #net.cuda()
  # testing wandb
  #wandb.watch(net, log_freq=100)
  #defining the criteria or our loss function 
  #criteria = nn.MSELoss()
  criteria = nn.SmoothL1Loss(beta=0.8)
  # defining our optimizer
  opt = Adam(net.parameters(),lr=CONFIG['LEARNING_RATE_ADAM'],weight_decay=CONFIG['WEIGHT_DECAY_ADAM'],amsgrad=True)
  #opt = torch.optim.SGD(net.parameters(), lr=CONFIG['LEARNING_RATE_ADAM'], momentum=0.9)
  #opt = Adam(net.parameters(),lr=CONFIG['LEARNING_RATE_ADAM'])
  # our scheduler
  scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=CONFIG['SCHEDULER_STEP_SIZE'], gamma=CONFIG['SCHEDULER_GAMMA'])
  # training and testing of the model 
  #history, predz = _fit_model_train_test(net,data_train,data_test,opt,criteria,DEVICE=device,n_iterations=EPOCHS)
  # testing every Nth batch
  history, predz = _fit_model_train_test_n_batches(net,
                                                   data_train,
                                                   data_test,
                                                   opt,
                                                   scheduler,
                                                   criteria,
                                                   device,
                                                   CONFIG['epochs'],
                                                   CONFIG['NUM_BATCHES_TO_TEST'])
  return history, predz