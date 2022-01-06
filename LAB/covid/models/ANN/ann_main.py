from ann_pipeline import make_pipeline

import wandb

def main():
	wandb.config = {"LEARNING_RATE_ADAM": 0.01,
	                "WEIGHT_DECAY_ADAM":1e-3,
	                "SCHEDULER_STEP_SIZE":20,
	                "SCHEDULER_GAMMA":0.90,
	                "epochs": 300,
	                "TRAIN_RATIO":0.70,
	                "NUM_BATCHES_TO_TEST":20,
	                "batch_size": 16,
	                "dataset":"Coswara-Cough",
	                "architectures":"ANN",
	                "opt":"SmoothL1Loss",
	                "AF":"ReLu",
	                "PATH":'./data/CovidCoughDataset.csv'}
	config = wandb.config
	return config 

if __name__ == "__main__":
	config = main()
	history, predictions = make_pipeline(config)
	