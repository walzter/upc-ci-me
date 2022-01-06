This is the repository for the final project for Computer Intelligence @ the UPC, Barcelona for the Master in AI


Dataset: Coswara-Covid-Cough


Models: ANN, CNN, RNN, Geometric Deep Neural Network (GDNN)


Framework: PyTorch 


Libraries: numpy, pandas, sklearn, pytorch, wandb, termcolor, librosa, 



ANN: 

The Artificial Neural Network is a simple model, with N hidden units separated in different layers, L. 
It first started with a shallow model of 4 Layers. This model was optimized until there was consistent overfitting on the test data. Once it reached this point, we began expanding the model until it reached satisfactory levels. 


Pipeline:

- Preprocessing: 
	- Audio (Wav) --> Feature Extraction using librosa 
	- Cleaning Data (removing NaNs/Null)
	- Removing Class #3 (corresponds to Unknown classification) 
	- Standardize the data with StandardScaler (sklearn)

- DataLoading:
	- Use dataloader class for PyTorch 
	- Create DataSet from torch.utils 

- Model:
	- Create a simple Sequential Linear Model with L layers
	- Activation Functions tested: ReLu, Sigmoid, LeakyReLU
	- Used Swish (Google's implementation of the Sigmoid)
	- Added Glorot Initiation to the first nn.Linear() layer 
	- Added He uniform and He normal to the aforementioned layer

- Training and Testing Loops:
	- Define Optimizer, Criteria (Loss Function) 
		- Optimizers tested: ADAM, SGD, RMSProp,
		- Criteria tested: MSE,SmoothL1Loss,BinaryCrossEntropy
	- Optional: define a scheduler
		- LR Scheduler from pytorch was used, it can be defined and passed to the training loop
			- Make sure to add: SCHEDULER.step() after the OPTIMIZER.step()
	- Iterate over the batches from DataSet
	- Every K batches test the model
	- Log the results to WandB 

- Hyperparameter Tuning: 
	- Adjusted parameters: 
		- ADAM: Learning Rate, WeightDecay, amsgrad =True/False
		- SGD: Learning Rate, Momentum
		- RMSProp: Learning Rate
		- Scheduler: Gamma, step_size

- Controlling the Parameters:
	- All parameters for this are controlled in the config object from WandB
		- This variable is in the "ann_pipeline.py" file


