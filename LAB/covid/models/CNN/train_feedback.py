import termcolor
from termcolor import colored
import torch


def train_check_loop(DATA_LOADER, DEVICE, MODEL, OPTIMIZER, CRITERIA, NUM_EPOCH_NOW):
    # TRAINING THE MODEL
    train_loss = 0.0
    train_correct = 0.0
    MODEL.train()
    for l, data in enumerate(DATA_LOADER, 0): # STEPS
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        inputs = inputs.permute(0, 3, 2, 1)
        inputs = inputs.float()
        # train mode
        #MODEL.train()
        # zeroing the gradients
        OPTIMIZER.zero_grad()
        # predictions
        outputs = MODEL(inputs)
        outputs = outputs.to(DEVICE)
        # Forward + Backwards + Optimization
        # backwards pass
        loss.backward()
        # stepping the optimizer
        OPTIMIZER.step()

        # get some feedback from the model
        _, prediction = torch.max(outputs.data, 1)
        correct = (prediction == labels).sum().item()
        train_correct += correct
        # getting the loss + accuracy
        train_loss += loss.item()
        test_acc = (train_correct/(len(DATA_LOADER.dataset)))
        if l % 100 == 0:
            train_text = f"\nTRAINING --> EPOCH: {NUM_EPOCH_NOW}, STEP: {l + 1}, LOSS: {train_loss/100}, ACCURACY: {test_acc*100}"
            print(colored(train_text, 'white', 'on_red', attrs=['bold']))
