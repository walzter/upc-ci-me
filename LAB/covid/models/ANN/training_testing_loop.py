import wandb
import torch
import termcolor
from termcolor import colored

from ann_utils import train_loop, test_loop

def _fit_model_train_test_n_batches(MODEL, DATA,TEST_DATA,OPTIMIZER,SCHEDULER, CRITERIA,DEVICE,EPOCHS,TEST_N_BATCHES):
  wandb.init(project="ci_lab_covid", entity="walzter")
  wandb.watch(MODEL, CRITERIA, log="all", log_freq=10)

  tracking_dict = {
                 'epoch':[],
                 'train_accuracy':[],
                 'train_loss':[],
                 'test_loss':[],
                 'test_accuracy':[],
                 'learning_rate_train':[],
                 'learning_rate_test':[],
                 'scheduled_learning_rate_train':[],
                 'scheduled_learning_rate_test':[],
                 }
  tracking_preds_dict = {'train_preds':[],
                         'test_preds':[]}
  for epoch in range(EPOCHS):
    train_loss = 0
    correct = 0
    for bidx, batch in enumerate(DATA):
      x_train, y_train = batch['input'], batch['output']
      x_train = x_train.to(DEVICE)
      y_train = y_train.to(DEVICE)
      y_train = y_train.unsqueeze(1)
      train_loss_val, train_preds = train_loop(MODEL, x_train, y_train,OPTIMIZER, CRITERIA)
      #metrics
      for idx, i in enumerate(train_preds):
        i = torch.round(i)
        if i == y_train[idx]:
          correct += 1
      train_acc = (correct/len(DATA.dataset))
      train_loss += train_loss_val.item()
      tracking_preds_dict['train_preds'].append(train_preds)
    

    # TESTING
    if epoch % TEST_N_BATCHES == 0:
      test_corr = 0
      test_loss = 0
      for tbdix, tbatch in enumerate(TEST_DATA):
        x_test, y_test = tbatch['input'],tbatch['output']
        x_test = x_test.to(DEVICE)
        y_test = y_test.to(DEVICE)
        y_test = y_test.unsqueeze(1)
        test_loss_val, test_preds,lr_test = test_loop(MODEL, x_test, y_test, OPTIMIZER, CRITERIA)
        for tdx, tr in enumerate(test_preds):
          tr = torch.round(tr)
          if tr == y_test[tdx]:
            test_corr += 1
        test_acc = (test_corr/(len(TEST_DATA.dataset)))
        test_loss += test_loss_val.item()
        tracking_preds_dict['test_preds'].append(test_preds)
    # history
    tracking_dict['epoch'].append(epoch)
    # training data
    tracking_dict['train_accuracy'].append(train_acc)
    tracking_dict['train_loss'].append(train_loss)
    #tracking_dict['learning_rate_train'].append(lr_train)
    #wandb
    wandb.log({"train_accuracy": train_acc})
    wandb.log({"train_loss": train_loss})
    #wandb.log({"learning_rate_train": lr_train})
    wandb.log({"scheduled_learning_rate_train": SCHEDULER.get_last_lr()[0]})

    # testing data
    tracking_dict['test_accuracy'].append(test_acc)
    tracking_dict['test_loss'].append(test_loss)
    tracking_dict['learning_rate_test'].append(lr_test)
    #wandb
    wandb.log({"test_accuracy": test_acc})
    wandb.log({"test_loss": test_loss})
    wandb.log({"learning_rate_test": lr_test})
    # debug
    if epoch % 5 == 0:
      text_train = f"\nTRAINING --> EPOCH: {epoch} ACCURACY: {train_acc*100}, LOSS: {train_loss}, LR: {OPTIMIZER.param_groups[0]['lr']}, SCHEDULED LR: {SCHEDULER.get_last_lr()[0]}"
      print(colored(text_train, 'white','on_red',attrs=['bold']))
    if (epoch != 0) and (epoch % TEST_N_BATCHES == 0):
      text_train = f"\nTESTING --> EPOCH: {epoch} ACCURACY: {test_acc*100}, LOSS: {test_loss}, LR: {lr_test}, SCHEDULED LR: {SCHEDULER.get_last_lr()[0]}"
      print(colored(text_train, 'green','on_grey',attrs=['bold']))

  return tracking_dict, tracking_preds_dict
