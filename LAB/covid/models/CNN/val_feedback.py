# feedback 
import termcolor
from termcolor import colored
import torch


def val_check_loop(VAL_DATA_LOADER, DEVICE, MODEL, OPTIMIZER, CRITERIA, NUM_EPOCH_NOW, N_BATCHES):
    if NUM_EPOCH_NOW % N_BATCHES == 0:
        val_loss = 0.0
        val_correct = 0.0
        for val_l, val_data in enumerate(VAL_DATA_LOADER, 0):
            vinputs, vlabels = val_data
            vinputs, vlabels = vinputs.to(DEVICE), vlabels.to(DEVICE)
            vinputs = vinputs.permute(0, 3, 2, 1)
            vinputs = vinputs.float()
		    # eval mode
            MODEL.eval()
		    # prediction
            val_pred = MODEL(vinputs)
		    # loss test
            loss_val = CRITERIA(val_pred, vlabels)
		    # zeroing the gradients
            OPTIMIZER.zero_grad()
		    # feedback 
            _, val_predz = torch.max(val_pred.data,1)
            vcorrect = (val_predz == vlabels).sum().item()
		    # getting the accuracy & loss
            val_correct+=vcorrect
            val_acc = (val_correct / (len(VAL_DATA_LOADER.dataset)))
            if val_l % 100 == 0:
                val_text = f"\nVALIDATION --> EPOCH: {NUM_EPOCH_NOW + 1}, STEP: {val_l + 1}, LOSS: {loss_val/100}, ACCURACY: {val_acc}"
                print(colored(val_text, 'red','on_yellow',attrs=['bold']))