# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prep_dataframe(PATH_TO_CSV):
  df = pd.read_csv(PATH_TO_CSV)
  # removing the unknowns
  rows_to_drop = df[df['num_label']==2].index.tolist()
  df = df.drop(rows_to_drop, axis=0)
  # dropping columns
  df = df.drop(['Unnamed: 0','ID','filename'],axis=1)
  X_vals, Y_vals = df.iloc[:,:-1].values, df.iloc[:,-1].values
  # scaling the values 
  scaler = StandardScaler().fit(X_vals)
  X_scaled = scaler.transform(X_vals)
  return X_scaled, Y_vals


# make the splits accordingly 
def train_test_val_split(X_values, Y_values,TRAIN_RATIO=0.70):
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
    X_train, X_test, Y_train, Y_test = train_test_split(X_values, Y_values, test_size= 1 - TRAIN_RATIO)

    # now we can get the validation set which is going to be 15% of the dataset
    # testing is now 15% of the original dataset 
    #X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size = TEST_RATIO / (TEST_RATIO + VALIDATION_RATIO))

    # grouping them for easier read
    training,testing = (X_train, Y_train),(X_test, Y_test)
            
    return training, testing


