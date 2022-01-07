import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def print_metrics(y_true, y_pred, tag):
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if y_true.ndim>1:
        y_true=y_true.ravel()
    if y_pred.ndim>1:
        y_pred=y_pred.ravel()
    val_acc = accuracy_score(y_true, y_pred)
    val_f1 = f1_score(y_true, y_pred)
    print(f"{tag} Acc: {val_acc} | {tag} F1: {val_f1}")
    return val_acc, val_f1
    
def prep_tab_df(PATH):
    # preparing the dataframe
    df = pd.read_csv(PATH)
    # removing the unknowns
    rows_to_drop = df[df['num_label']==2].index.tolist()
    df = df.drop(rows_to_drop, axis=0)
    # dropping columns
    df = df.drop(['Unnamed: 0','ID','filename'],axis=1)# splitting the data 
    #(X_train, y_train), (X_test, y_test) = train_test_val_split(X_scaled, y_vals, TRAIN_RATIO=TRAIN_RATIO)
    num_cols = [cols for cols in df.columns if cols!='num_label']
    return df, num_cols