'''
A tool kit for port analysis.
'''

import pandas as pd
import numpy as np
import seaborn as sns
import os
from matplotlib import pyplot as plt
import datetime as dt

data_path = os.path.dirname(os.getcwd())
data_path_in = os.path.join(data_path, 'introToFinLocal','data', 'output_lgbm.csv')

def sharpe(data: pd.DataFrame, col: str):
    '''
    Return the annualized sharpe ratio of a return rate column.
    '''
    return data[col].mean() / data[col].std() * np.sqrt(12)

def cum_ret(data: pd.DataFrame, col: str):
    '''
    Return the cumulative return of a return rate column.
    '''
    cumulative_return = (data[col] + 1).cumprod() - 1

    return cumulative_return

def benchmark(data: pd.DataFrame, col: str):

    # 月度股票平均收益率
    data_grouped = data.groupby(level=0, group_keys=False)[col].mean()
    bm = (data_grouped+1).cumprod()-1
    
    return bm

def max_1m_loss(data: pd.DataFrame, col: str):
    '''
    Return the max 1 month loss of a return rate column.
    '''
    return data[col].min()

def max_drawdown(data: pd.DataFrame, col: str):
    '''
    Return the max drawdown of a return rate column.
    '''
    log_ret = np.log(data[col] + 1)
    log_ret = log_ret.cumsum()
    rolling_peak = np.maximum.accumulate(log_ret)
    drawdowns = rolling_peak - log_ret
    return drawdowns.max()

def r2(data: pd.DataFrame, real: str, pred: str):
    '''
    Return the OOS R2 of real values and predicted values.
    '''
    res = 1 - np.sum(
        np.square(data[real] - data[pred])
    ) / np.sum(np.square(data[real]))
    return res

def long_short(data: pd.DataFrame, col: str, k: int = 80):
    '''
    Weighted long-short strategy. Rank stocks based on the absolute value of predicted returns,
    then assign weights based on the sign of the predicted return (positive for long, negative for short).
    '''
    # 按预测收益率的绝对值进行排序
    data = data.groupby(level=0, group_keys=False).apply(
        lambda x: x.reindex(x[col].abs().sort_values(ascending=False).index)
    )
    
    grouped = data.groupby(level=0, group_keys=False)

    # 多头和空头的权重
    def weighted_mean(x:pd.DataFrame, is_long=True):
        # 选取绝对值排名前k的股票
        top_k = x.iloc[:k].copy()
        
        # 根据原始预测值的正负进行还原，确定多头和空头
        if is_long:
            long = top_k[top_k[col] > 0].copy()  # 选择预测收益率为正的股票
            return long['stock_exret'].mean() if long.size > 0 else 0
        else:
            short = top_k[top_k[col] < 0].copy()  # 选择预测收益率为负的股票
            return short['stock_exret'].mean() if short.size > 0 else 0

    # 计算多头和空头收益
    best_group_ret = grouped.apply(lambda x: weighted_mean(x, is_long=True))  # 多头
    worst_group_ret = grouped.apply(lambda x: weighted_mean(x, is_long=False))  # 空头
    
    port = best_group_ret - worst_group_ret  # 多空组合收益
    port.name = 'stock_exret'  # 返回的收益列名
    return port

if __name__ == '__main__':
    data = pd.read_csv(data_path_in)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    
    # st = dt.datetime(2010,1,1)
    # et = st + pd.DateOffset(years=10)
    # data = data.loc[st:et]

    col = 'lgbm'
    port_ret = long_short(data, col).to_frame()

    col = 'stock_exret'
    print('Annualized Sharpe:', sharpe(port_ret, col))
    #print('Cummulative return:', cum_ret(port_ret, col))
    print('Max 1m loss:', max_1m_loss(port_ret, col))
    print('Max Drawdown:', max_drawdown(port_ret, col))
    cr = cum_ret(port_ret, col)

    bm = benchmark(data,col)

    plt.figure(figsize=(10, 6))
    plt.plot(bm, label=f'benchmark({col})', color='blue')
    plt.plot(cr, label=f'cumulative return({col})', color='red')
    plt.title('benchmark and cumulative return Over Time')
    plt.xlabel('Time')
    plt.ylabel('stock excess return')
    plt.legend()
    plt.grid(True)
    plt.show()