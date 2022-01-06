import torch.nn.functional as F
# activation function

def swish(x):
  '''
  Google's implementation of the sigmoidal 
  swish(x) := x * sigmoid(Bx) = x/ 1 + e^(-Bx)
  '''
  return x * F.sigmoid(x)


# defining a training loop 
def train_loop(model, X, Y,optimizer, criterion):
  # make sure the gradients are zero 
  model.train()
  # optimizer grad to zero 
  optimizer.zero_grad()
  # predicting the output
  output = model(X.float())
  # using the defined loss criteria 
  loss = criterion(output, Y.float())
  #propagate backwards 
  loss.backward()
  #using the optimizer to step the gradients 
  # w_i += ∆(∂w/∂b)
  optimizer.step()
  # step the scheduler
  #scheduler.step()
  # return the loss and the output 
  #learning_rate_train = optimizer.param_groups[0]['lr']
  return loss, output #,learning_rate_train

def test_loop(model, X, Y, optimizer, criterion):
  # make sure the gradients are zero 
  model.eval()
  # predicting the output
  prediction_test = model(X.float())
  # using the defined loss criteria 
  loss_test = criterion(prediction_test, Y.float())
  #propagate backwards 
  #using the optimizer to step the gradients 
  # w_i += ∆(∂w/∂b)
  optimizer.zero_grad()
  #scheduler to zero
  #SCHEDULER.step()
  # return the loss and the output 
  learning_rate_test = optimizer.param_groups[0]['lr']
  return loss_test, prediction_test, learning_rate_test
