import pytorch_tabular
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from sklearn.preprocessing import PowerTransformer
from TAB_utils import print_metrics
import itertools
import gc


from sklearn.model_selection import train_test_split


def MakeCategoryEmbeddingModel(target_col, num_cols,batch_size=1024,epochs=100,auto_find_lr=True,to_use_gpu=None,task='classification',layers='1024-512-512',act_f = 'LeakyReLU',lr = 1e-3):

    data_config = DataConfig(
        target=[target_col], #target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
        continuous_cols=num_cols,num_workers=4,
    )
    trainer_config = TrainerConfig(
        auto_lr_find=auto_find_lr, # Runs the LRFinder to automatically derive a learning rate
        batch_size=batch_size,
        max_epochs=epochs,
        gpus=to_use_gpu,  #index of the GPU to use. -1 means all available GPUs, None, means CPU
    )
    optimizer_config = OptimizerConfig()

    model_config = CategoryEmbeddingModelConfig(
        task=task,
        layers=layers,  # Number of nodes in each layer
        activation=act_f, # Activation between each layers
        learning_rate = lr
    )

    name = f"CategoryEmbeddingModel_{batch_size}_{epochs}_{layers}_{act_f}_{lr}"
    
    #experiment_config = ExperimentConfig(project_name="CategoricalEmbeddingModel",run_name=name, log_target="wandb", log_logits=True)
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        #experiment_config=experiment_config
    )
    
    return tabular_model, name

def test_model(MODEL,DATAFRAME,RANDOM_STATE=42, save_model=False, model_name=None):
    X_train, X_test = train_test_split(DATAFRAME, random_state=RANDOM_STATE)
    X_train, X_val = train_test_split(X_train, random_state=RANDOM_STATE)
    # fitting the model 
    MODEL.fit(train=X_train,
              validation=X_val)
    # evaluating the model 
    res = MODEL.evaluate(X_test)
    # predicting 
    preds = MODEL.predict(X_test)
    # printing some metrics 
    val_acc, val_f1 = print_metrics(X_test['num_label'], preds['prediction'],tag='Holdout')
    
    if save_model:
        # we can even save a model 
        MODEL.save_model(f"./model_check/{model_name}")
        return res, preds, val_acc, val_f1
    else:
        return res, preds,val_acc, val_f1


def custom_train_loop(DATAFRAME, TARGET_COL, NUM_COL,BATCHES, NUM_EPOCHS, NUM_LAYERS, ACT_FUNCS,LEARNING_RATES):
    MOTHER_LIST = [BATCHES, NUM_EPOCHS, NUM_LAYERS, ACT_FUNCS,LEARNING_RATES]
    COMBS = list(itertools.product(*MOTHER_LIST))
    res_dict = dict()
    for PERM in COMBS:
        CEM_model,name = MakeCategoryEmbeddingModel(TARGET_COL,
                                       NUM_COL,
                                       batch_size=PERM[0],
                                       epochs=PERM[1],
                                       auto_find_lr=True,
                                       to_use_gpu=None,
                                       task='classification',
                                       layers=PERM[2],
                                       act_f = PERM[3],
                                       lr = PERM[4])
        results, predictions,val_acc, val_f1 = test_model(CEM_model,DATAFRAME,RANDOM_STATE=42, save_model=False, model_name=None)
        res_dict[name] = {"Val_acc":val_acc,"val_f1":val_f1}
        print(f"\n--------Finished PERMUATIONS {PERM}------------------\n")
        gc.collect()
        print(res_dict)
        
    return res_dict
        