import torch 
import torch.nn as nn 
import torchvision 

# CNN model 

class ConvNet(nn.Module):

	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d()
		self.conv2 = nn.Conv2d()
		self.conv3 = nn.Conv2d()

	def forward(self, x):

		x = ...
		x = ...
		x = ...
		x = ...
		x = ...
		x = ...
	return x 
