# imports
from sklearn.model_selection import train_test_split
from cnn_data_loader import CovidCoughDatasetSpectrograms
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader

# splits


def train_test_val_split(X_values, Y_values, TRAIN_RATIO=0.70):
    '''
    Args:
        X_values (Matrix): Features 
        Y_values (Column Vector): Labels
        TRAIN_RATIO (Float): % to split the Data into training - default = 0.70 or 70%
        VALIDATION_RATIO (Float): % to split the data into validation - default = 0.15 or 15%
        TEST_RATIO (Float): % to split the data into testing - default = 0.15 or 15%
    Output:
        Splitting the X_values and the Y_values according to: 
            - train_ratio
            - validation_ratio 
            - test_ratio 
    '''

    # This will use the percentage of the test_size approx to train the values
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_values, Y_values, test_size=1 - TRAIN_RATIO)

    # now we can get the validation set which is going to be 15% of the dataset
    # testing is now 15% of the original dataset
    #X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size = TEST_RATIO / (TEST_RATIO + VALIDATION_RATIO))

    # grouping them for easier read
    training, testing = (X_train, Y_train), (X_test, Y_test)

    return training, testing


# loading
def load_data_split(PATH_TO_FILE, TRAIN_RATIO, VALIDATION_RATIO, BATCH_SIZE, TO_SHUFFLE=True):
    '''
    Takes in a pandas.DataFrame with two columns (IMG_PATH and LABEL)

    input:type:DATAFRAME: pd.DataFrame
    input:type:STRING: PATH to the file with the data 
    output:type: numpy.array with text and labels splitted for training, testing and validation
    '''

    df = pd.read_csv(PATH_TO_FILE)
    df = df.drop('Unnamed: 0',axis=1)
    df['PATH'] = df['PATH'].astype(str)
    X, Y = df.iloc[:, 0].values, df.iloc[:, 1].values

    # splitting
    (X_train, y_train), (X_test, y_test) = train_test_val_split(X, Y, TRAIN_RATIO=TRAIN_RATIO)
    (X_train, y_train), (X_val, y_val) = train_test_val_split(X_train, y_train, TRAIN_RATIO=VALIDATION_RATIO)

    # making the DataSets
    # Training
    train_dataset = CovidCoughDatasetSpectrograms(y_train, X_train)
    # Testing
    test_dataset = CovidCoughDatasetSpectrograms(y_test, X_test)
    # Validation
    val_dataset = CovidCoughDatasetSpectrograms(y_val, X_val)

    # Making the loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=TO_SHUFFLE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=TO_SHUFFLE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=TO_SHUFFLE)
    # datasets and dataloaders 
    datasets = (train_dataset, test_dataset, val_dataset)
    dataloaders = (train_loader, test_loader, val_loader)

    return datasets, dataloaders