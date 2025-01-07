'''
Lambdarank.
'''
import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import datetime as dt
from tqdm import tqdm
import random
import PortAnalysis as pa
import AlphaMiner_v2 as am
from Factory import Factory

def lgbm_reg(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict
    ):

    group_train = y_train.groupby(level=0).value_counts(sort=False).values
    group_val = y_val.groupby(level=0).value_counts(sort=False).values

    lgb_train = lgb.Dataset(X_train, y_train, group=group_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, group=group_val)

    gbm = lgb.train(
        params, lgb_train, num_boost_round=100,
        valid_sets=lgb_val
    )

    return gbm

if __name__ == '__main__':

    # NOTE: set working directory
    data_path = 'C:\\CODES\\CODE_PYTHON\\stock_sample.csv'

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'learning_rate': 1e-4,
        'max_depth': -1,
        'min_data_in_leaf': 500,
        'verbose': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1
    }

    k = 10
    fac = Factory(data_path, k, True)
    data = fac.load_data(data_path)

    # out stores the prediction results of each year, 
    # starting from 2010, and finally concatenate them.
    out = []
    for i in tqdm(range(2008, 2022)):
        val_st = dt.datetime(i, 1, 1)
        X_train, X_val, y_train, y_val = fac.load_lgbm_dataset_original(data, val_st)
        gbm = lgbm_reg(X_train, y_train, X_val, y_val, params)

        test_st = dt.datetime(i + 2, 1, 1)
        test_et = dt.datetime(i + 2, 12, 31)

        y_real_cols = ['permno', 'stock_exret', 'stock_exret_rank']
        X_test = data.loc[test_st:test_et].drop(columns=y_real_cols)
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        
        output = data.loc[test_st:test_et, y_real_cols].copy()
        output['lgbm'] = y_pred
        output_grouped = output.groupby(level=0)['lgbm']
        output['lgbm_rank'] = (output_grouped.rank(method='min') - 1) // (output_grouped.size() // k + 1)

        # Instead of R2, we choose Spearman Corr to measure accuracy.
        sp_corr = output['lgbm_rank'].corr(output['stock_exret_rank'], method='spearman')
        print('Spearman Corr:', sp_corr)

        out.append(output)

    out = pd.concat(out)

    print('##################################')

    tot_corr = out['lgbm_rank'].corr(out['stock_exret_rank'], method='spearman')
    print('Rank Corr of all test sets:', tot_corr)

    best_group = out[out['lgbm_rank'] >= k - 1]
    worst_group = out[out['lgbm_rank'] <= 1]
    best_corr = best_group['stock_exret_rank'].corr(best_group['lgbm_rank'], method='spearman')
    worst_corr = worst_group['stock_exret_rank'].corr(worst_group['lgbm_rank'], method='spearman')
    print('The prediction accuracy of best stocks:', best_corr)
    print('The prediction accuracy of worst stocks:', worst_corr)

    best_ret = best_group['stock_exret'].mean()
    worst_ret = worst_group['stock_exret'].mean()
    print('Equal weight portfolio of predicted best stocks:', best_ret)
    print('Equal weight portfolio of predicted worst stocks:', worst_ret)

    out.to_csv('output_lgbm.csv')