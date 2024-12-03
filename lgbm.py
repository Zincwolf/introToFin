import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import winsorize
import pandas as pd
import numpy as np
import os
import datetime as dt
from tqdm import tqdm

# 12.3
# TODO: 先试试不引入行业变量，直接用lgbm做回归
# TODO: 引入行业变量

# NOTE: set working directory
# 注意树模型的输入不需要标准化
data_path = os.path.dirname(os.getcwd())
data_path = os.path.join(data_path, 'introToFin_utils', 'stock_sample.csv')

def load_data(data_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    # The columns left: index, permno, stock_exret and 147 factors
    data.drop(
        columns=['year', 'month', 'stock_ticker', 'comp_name'],
        inplace=True
    )
    return data

def split_data(
        data: pd.DataFrame,
        val_st: dt.datetime
    ):
    '''
    Split the data into train and valid sets. 
    By default, the valid set is 2 years long.
    '''
    val_et = val_st + pd.DateOffset(years=2) - dt.timedelta(days=1)
    len_val = len(data.loc[val_st:val_et])
    features = data.loc[:val_et].drop(columns=['stock_exret','permno'])
    labels = data.loc[:val_et, 'stock_exret']
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=len_val, shuffle=False
    )
    return X_train, X_val, y_train, y_val

# 使用lightgbm做横截面回归
def lgbm_reg(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ):
    # 将数据转换为lgbm的数据格式
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # 设置参数
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        # 'num_leaves': 31,
        'max_depth': 2,
        'n_estimators': 80,
        'min_data_in_leaf': 28,
        # 'min_sum_hessian_in_leaf': 20,
        'learning_rate': 0.06,
        'num_iterations': 100,
        'feature_fraction': 0.95,
        'bagging_fraction': 0.8,
        'bagging_freq': 10,
        'verbose': 1,
        'lambda_l1': 0.02,
        'lambda_l2': 0.02,
        'early_stopping_round': 50,
    }

    # 训练模型
    gbm = lgb.train(
        params, lgb_train, num_boost_round=100,
        valid_sets=lgb_val
    )

    # 预测
    y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)

    # 评估
    mse = mean_squared_error(y_val, y_pred)
    print('Loss (MSE):', mse)

    return gbm

if __name__ == '__main__':
    data = load_data(data_path)
    # out是每一年的预测结果，从2010年开始，最后concat起来
    out = []
    for i in tqdm(range(2008, 2022)):
        # 最后一个验证集是2021年和2022年
        val_st = dt.datetime(i, 1, 1)
        X_train, X_val, y_train, y_val = split_data(data, val_st)
        gbm = lgbm_reg(X_train, y_train, X_val, y_val)

        test_st = dt.datetime(i + 2, 1, 1)
        test_et = dt.datetime(i + 2, 12, 31)
        X_test = data.loc[test_st:test_et].drop(columns=['stock_exret','permno'])
        y_pred = gbm.predict(X_test)
        
        output = data.loc[test_st:test_et, ['permno','stock_exret']].copy()
        output['lgbm'] = y_pred
        out.append(output)

    out = pd.concat(out)
    out.to_csv('output_lgbm.csv')