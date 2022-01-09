import termcolor
from termcolor import colored
import torch

def test_check_loop(VAL_DATA_LOADER,DEVICE,MODEL,OPTIMIZER,CRITERIA, NUM_EPOCH_NOW, N_BATCHES):
	if EPOCH % N_BATCHES == 0: 
    val_loss = 0.0
    val_correct = 0.0
    for val_l, val_data in enumerate(VAL_DATA_LOADER,0):
      # putting it into eval mode 
      with torch.no_grad():
	      vinputs, vlabels = val_data
	      vinputs, vlabels = vinputs.to(DEVICE),vlabels.to(DEVICE)
	      vinputs = vinputs.permute(0,3,2,1)
	      vinputs = vinputs.float() 
	      # eval mode
	      net.eval()
	      # prediction
	      val_pred = net(vinputs)
	      # loss test
	      loss_val = CRITERIA(val_pred, vlabels)
	      # zeroing the gradients
	      OPTIMIZER.zero_grad()
	      # feedback 
	      _, val_predz = torch.max(test_pred.data,1)
	      tcorrect = (val_predz == tlabels).sum().item()
	      # getting the accuracy & loss
	      val_correct+=tcorrect
	      val_acc = (val_correct / (len(VAL_DATA_LOADER.dataset)))
	      if val_l % 100 == 0:
	        test_text = f"\nVALIDATION --> EPOCH: {NUM_EPOCH_NOW + 1}, STEP: {l + 1}, LOSS: {loss_val/100}, ACCURACY: {val_acc}"
	        print(colored(test_text, 'red','yellow',attrs=['bold']))