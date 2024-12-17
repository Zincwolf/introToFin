
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats.mstats import winsorize
import pandas as pd
import numpy as np
import os
import datetime as dt

# params for lgbm WITH feature select
# param_df = pd.DataFrame({
#     'boosting_type': ['gbdt','gbdt','gbdt','gbdt','gbdt','gbdt','gbdt','gbdt','gbdt','gbdt','gbdt','gbdt','gbdt', 'gbdt'],
#     'objective': ['regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression'],
#     'metric': ['rmse', 'rmse', 'rmse', 'rmse', 'rmse', 'rmse', 'rmse', 'rmse', 'rmse', 'rmse', 'rmse', 'rmse', 'rmse', 'rmse', ],
#     'num_leaves': [10, 5, 9, 5, 5, 10, 5, 5, 5, 5, 5, 5, 5, 5],
#     'max_depth': [4, 3, 3, 3, 3, 4, 3, 3, 3, 3, 4, 3, 3, 3],
#     'n_estimators': [200, 250, 350, 300, 1500, 150, 1500, 1500, 1500, 1500, 1500, 1500, 1500,1500],
#     'min_data_in_leaf': [5, 15, 6, 15, 15, 5, 15, 15, 15, 30, 30, 15, 15, 15],
#     'learning_rate': [0.1,0.15,0.07,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.03,0.05,0.05,0.05],# 调大学习率
#     'feature_fraction': [0.5, 0.3, 0.5, 0.5, 0.95, 0.5, 0.95, 0.95, 0.95, 0.9, 0.95, 0.95, 0.95, 0.95],
#     'bagging_fraction': [0.8, 1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
#     'bagging_freq': [1, 5, 3, 3, 10, 10, 10, 10, 10, 10, 5, 10, 10, 10],
#     'verbose': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     'lambda_l1': [0.0, 0.0, 1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1],
#     'lambda_l2': [0.0, 0.0, 3, 0.0, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.0, 0.23, 0.23, 0.23],
#     'early_stopping_round': [50, 150, 150, 200, 200, 50, 200, 200, 200, 200, 250, 200, 200, 200]
# })

class Params:
    '''
    get the params of the light gbm
    '''
    def __init__(self):
        pass

    def get(i: int):
        if i < len(param_df):
            return param_df.iloc[i].to_dict()
        else:
            raise IndexError(f"Index {i} is out of range. Maximum allowed index is {len(param_df) - 1}.")


param_df = pd.DataFrame({
    'boosting_type': ['gbdt','gbdt','gbdt','gbdt','gbdt','gbdt','gbdt','gbdt','gbdt','gbdt','gbdt','gbdt','gbdt', 'gbdt'],
    'objective': ['regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression'],
    'metric': ['rmse', 'rmse', 'rmse', 'rmse', 'rmse', 'rmse', 'rmse', 'rmse', 'rmse', 'rmse', 'rmse', 'rmse', 'rmse', 'rmse', ],
    'num_leaves': [6,10,6,6,10,10,6,6,6,6,6,6,6,6],
    'max_depth': [3,4,3,3,4,4,3,3,3,3,3,3,3,3],
    'n_estimators': [85, 185, 85, 85, 185, 185, 85, 85, 85, 85, 85, 85, 85, 85],
    'min_data_in_leaf': [30,10,30,30,10,10,30,30,30,30,30,30,30,30],
    'learning_rate': [0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],# 调大学习率
    'feature_fraction': [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
    'bagging_fraction': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    'bagging_freq': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    'verbose': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'lambda_l1': [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
    'lambda_l2': [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25],
    'early_stopping_round': [50,100,50,50,100,100,50,50,50,50,50,50,50,50]
})
