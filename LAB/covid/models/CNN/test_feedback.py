import termcolor
from termcolor import colored

def test_check_loop(TEST_DATA_LOADER,DEVICE,MODEL,OPTIMIZER,CRITERIA, NUM_EPOCH_NOW, N_BATCHES):
# TESTING
  if NUM_EPOCH_NOW % N_BATCHES == 0: 
    test_loss = 0.0
    test_correct = 0.0
    for test_l, test_data in enumerate(TEST_DATA_LOADER,0):
      # test inputs and labels 
      tinputs, tlabels = test_data
      tinputs, tlabels = tinputs.to(DEVICE), tlabels.to(DEVICE)
      tinputs = tinputs.permute(0,3,2,1)
      tinputs = tinputs.float() 
      # putting it into eval mode 
      MODEL.eval()
      # prediction
      test_pred = net(tinputs)
      test_pred = test_pred.to(DEVICE)
      # loss test
      loss_test = CRITERIA(test_pred, tlabels)
      # zeroing the gradients
      OPTIMIZER.zero_grad()
      # feedback 
      _, test_predz = torch.max(test_pred.data,1)
      tcorrect = (test_predz == tlabels).sum().item()
      # getting the accuracy & loss
      test_correct+=tcorrect
      test_acc = (test_correct / (len(TEST_DATA_LOADER.dataset)))
      if test_l % 100 == 0:
        test_text = f"\nTESTING --> EPOCH: {NUM_EPOCH_NOW + 1}, STEP: {l + 1}, LOSS: {test_loss/100}, ACCURACY: {test_acc}"
        print(colored(test_text, 'green','on_grey',attrs=['bold']))