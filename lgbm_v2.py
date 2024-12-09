import lightgbm as lgb
from scipy.stats.mstats import winsorize
import pandas as pd
import numpy as np
import os
import datetime as dt
from tqdm import tqdm
import random

# 12.8 试一下把rank作为因变量去回归

# NOTE: set working directory
# 注意树模型的输入不需要标准化
# 但是不清楚lambdarank是否需要
data_path = os.path.dirname(os.getcwd())
data_path = os.path.join(data_path, 'introToFin_utils', 'stock_sample.csv')

black_list = []
# with open('/Users/znw/Code_python/introToFin_utils/black_list.txt') as f:
#     black_list = f.read().splitlines()
trivials = ['year', 'month', 'stock_ticker', 'comp_name']
k = 10

def load_data(
        data_path: str, 
        is_y_rank: bool = False
    ) -> pd.DataFrame:
    '''
    Load data from stock examples.
    NOTE: We do not clean data here.
    Args:
        is_y_rank: transform stock returns to ranks or not.
    '''
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    # The columns left: index, permno, stock_exret and factors
    data.drop(
        columns=trivials + black_list,
        inplace=True
    )

    # Group by months, ranks by high (1) to low (N). (ascending=False)
    # 每个月rank会很大，有上千个，考虑转化为“重要性标签”，也就是将排名划分为k组梯队
    if is_y_rank:
        # k = 10
        grouped = data.groupby(level=0)['stock_exret']
        data['stock_exret_rank'] = (grouped.rank(method='min') - 1) // (grouped.size() // k)

    return data

def split_data(
        data: pd.DataFrame,
        val_st: dt.datetime,
        val_year: int = 2,
        is_y_rank: bool = False
    ):
    '''
    Split the data into train and valid sets. 
    Data are shuffled in train set and valid set respectively.

    Args:
        val_st: the start time (month) of valid set. 
        val_year: the number of years in the valid set.
        is_y_rank: whether the ys are ranks or not.
    '''

    def shuffle_data(data: pd.DataFrame):
        # Shuffle the index for random batches
        # 日期有重复，所以先要换为自然数索引
        data.reset_index(inplace=True)
        rand_idx = data.index.to_list()
        random.shuffle(rand_idx)
        data = data.reindex(pd.Index(rand_idx))
        data.set_index('date', inplace=True)
        return data

    val_et = val_st + pd.DateOffset(years=val_year, days=-1)
    val = shuffle_data(data[val_st:val_et])
    train = shuffle_data(data[:val_st - pd.DateOffset(months=1)])

    # NOTE: We temporarily do not consider permno, or industry information
    # when training.
    y_real_cols = ['permno', 'stock_exret']
    if is_y_rank:
        y_real_cols += ['stock_exret_rank']
    X_train = train.drop(columns=y_real_cols)
    X_val = val.drop(columns=y_real_cols)

    if is_y_rank:
        y_train, y_val = train['stock_exret_rank'], val['stock_exret_rank']
    else:
        y_train, y_val = train['stock_exret'], val['stock_exret']

    return X_train, X_val, y_train, y_val

# 使用lightgbm做横截面回归
def lgbm_reg(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict,
        is_y_rank: bool = False
    ):
    # 对排名进行回归，还需要提供分组数据，也就是每个月有多少股

    group_train, group_val = None, None
    if is_y_rank:
        group_train = y_train.groupby(level=0).value_counts(sort=False).values
        group_val = y_val.groupby(level=0).value_counts(sort=False).values

    lgb_train = lgb.Dataset(X_train, y_train, group=group_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, group=group_val)

    # 训练模型
    gbm = lgb.train(
        params, lgb_train, num_boost_round=100,
        valid_sets=lgb_val
    )

    return gbm

if __name__ == '__main__':

    params = {
        'objective': 'lambdarank',
        'metric': 'map',
        'learning_rate': 0.1,
        'max_depth': 10,
        'n_estimators': 50,
        'verbose': 1,
    }

    data = load_data(data_path, True)
    # out是每一年的预测结果，从2010年开始，最后concat起来
    out = []
    for i in tqdm(range(2008, 2022)):
        # 最后一个验证集是2021年和2022年
        val_st = dt.datetime(i, 1, 1)
        X_train, X_val, y_train, y_val = split_data(data, val_st, is_y_rank=True)
        gbm = lgbm_reg(X_train, y_train, X_val, y_val, params, True)

        test_st = dt.datetime(i + 2, 1, 1)
        test_et = dt.datetime(i + 2, 12, 31)

        y_real_cols = ['permno', 'stock_exret', 'stock_exret_rank']
        X_test = data.loc[test_st:test_et].drop(columns=y_real_cols)
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        
        output = data.loc[test_st:test_et, y_real_cols].copy()
        output['lgbm'] = y_pred
        output_grouped = output.groupby(level=0)['lgbm']
        output['lgbm_rank'] = (output_grouped.rank(method='min') - 1) // (output_grouped.size() // k)

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