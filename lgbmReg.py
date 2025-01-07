import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import datetime as dt
from lgbm_params import Params
from tqdm import tqdm
from Factory import Factory

def load_data(data_path: str) -> pd.DataFrame:
    '''
    Load stock data from a CSV file, convert date column to datetime format, and clean missing values.
    
    Args:
        data_path (str): Path to the stock sample CSV file.
    
    Returns:
        pd.DataFrame: Processed stock data with missing values filled by median.
    '''
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    # Drop unused columns
    data.drop(
        columns=['year', 'month', 'stock_ticker', 'comp_name'],
        inplace=True
    )
    # Fill missing values with group-wise median
    data = data.groupby(level=0).transform(lambda x: x.fillna(x.median()))
    return data

def oos_r2(y_true, y_pred) -> float:
    '''
    Compute Out-of-Sample R^2 score.
    
    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
    
    Returns:
        float: OOS R^2 score.
    '''
    sst = np.sum(np.square(y_true))
    sstr = np.sum(np.square(y_true - y_pred))
    return 1 - (sstr / sst)

def lgbm_reg(X_train, y_train, X_val, y_val, params):
    '''
    Train a LightGBM regression model and evaluate it using OOS R^2 on validation data.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        params (dict): Parameters for LightGBM model.
    
    Returns:
        tuple: (Trained model, OOS R^2 score, List of training features)
    '''
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    gbm = lgb.train(
        params, lgb_train,
        valid_sets=lgb_val
    )

    y_pred_val = gbm.predict(X_val)
    OOS_R2_val = oos_r2(y_val, y_pred_val)

    train_features = X_train.columns
    return gbm, OOS_R2_val, train_features

def split_data(
    data: pd.DataFrame,
    val_st: dt.datetime
):
    '''
    Split the data into train and validation sets.
    
    Args:
        data (pd.DataFrame): Stock data.
        val_st (datetime): Start date of validation set (2 years duration).
    
    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    '''
    val_et = val_st + pd.DateOffset(years=2) - dt.timedelta(days=1)
    train_st = dt.datetime(2000,1,1)
    len_val = len(data.loc[val_st:val_et])
    
    # Select features and labels

    # feature selection proves to be useless here
    # features = selector.rankIC_method(data.loc[train_st:val_et],num=IC_n)
    # features = selector.ANOVA_method(features,num=anova_n).drop(columns=['stock_exret','permno'])
    # print(features.columns)

    features = data.loc[train_st:val_et].drop(columns=['stock_exret','permno'])
    labels = data.loc[train_st:val_et, 'stock_exret']

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=len_val, shuffle=False
    )
    return X_train, X_val, y_train, y_val

if __name__ == '__main__':
    
    # Load and process the data
    # NOTE: Set your data path here
    data_path = os.path.dirname(os.getcwd())
    data_path_in = 'C:\\CODES\\CODE_PYTHON\\stock_sample.csv'
    data_path_out = os.path.join(data_path, 'output_lgbm.csv')
    factory = Factory(data_path_in)
    data = factory.load_data()
    
    out = []  # List to store yearly predictions
    OOS_R2_val = []  # List to store validation OOS R^2

    for i in tqdm(range(2008, 2022)):
        # Define validation period start date
        val_st = dt.datetime(i, 1, 1)
        X_train, X_val, y_train, y_val = factory.split_data(data, val_st)
        
        # Train the model and calculate validation R^2
        gbm, OOS_R2_val_value, this_features = lgbm_reg(X_train, y_train, X_val, y_val, Params.get(i-2008))
        OOS_R2_val.append({'year': i, 'oos_r2': OOS_R2_val_value})

        # Test on future year (2 years after validation start)
        test_st = dt.datetime(i + 2, 1, 1)
        test_et = dt.datetime(i + 2, 12, 31)
        X_test = data.loc[test_st:test_et].drop(columns=['stock_exret','permno'])
        X_test = X_test[this_features]
        y_test = data.loc[test_st:test_et, 'stock_exret']
        y_pred = gbm.predict(X_test)

        # Store predictions and actual returns
        output = data.loc[test_st:test_et, ['permno','stock_exret']].copy()
        output['lgbm'] = y_pred
        output['group'] = output.index.to_period('M')
        out.append(output)

    # Combine yearly predictions
    out = pd.concat(out)
    OOS_R2_val = pd.DataFrame(OOS_R2_val).set_index('year')

    # Compute yearly test OOS R^2
    out['year'] = out.index.year
    OOS_R2_test = out.groupby('year').apply(lambda x: oos_r2(x['stock_exret'], x['lgbm']))
    OOS_R2_test = pd.DataFrame(OOS_R2_test, columns=['oos_r2'])

    # Print validation OOS R^2
    print(OOS_R2_val)
    print('AVRG Validation OOS R2:', OOS_R2_val['oos_r2'].mean())

    # Print test set OOS R^2
    print(OOS_R2_test['oos_r2'])
    print('AVRG Test OOS R2:', OOS_R2_test['oos_r2'].mean())

    # Compute overall OOS R^2
    OOS_R2_all = oos_r2(out['stock_exret'], out['lgbm'])
    print('OOS R2:', OOS_R2_all)

    # Save predictions to CSV
    out.to_csv(data_path_out)
