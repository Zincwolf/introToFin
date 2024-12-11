'''
A tool kit for port analysis.
'''

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Union, List

def sharpe(data: pd.DataFrame, cols: Union[str, List[str]]):
    '''
    Return the annualized sharpe ratio of a return rate column.
    '''
    return data[cols].mean() / data[cols].std() * np.sqrt(12)

def cum_ret(data: pd.DataFrame, cols: Union[str, List[str]]):
    '''
    Return the cumulative return of a return rate column.
    '''
    temp = data[cols] + 1
    temp = temp.cumprod() - 1
    return temp

def inv_cum_ret(data: pd.DataFrame, cols: Union[str, List[str]]):
    '''
    An inverse function of `cum_ret`.
    '''
    temp = data[cols] + 1
    temp = temp / temp.shift(1) - 1
    
    first = data.index[0]
    temp.iloc[0] = data.loc[first, cols]

    return temp

def max_1m_loss(data: pd.DataFrame, cols: Union[str, List[str]]):
    '''
    Return the max 1 month loss of a return rate column.
    '''
    return data[cols].min()

def max_drawdown(data: pd.DataFrame, cols: Union[str, List[str]]):
    '''
    Return the max drawdown of a return rate column.
    '''
    log_ret = np.log(data[cols] + 1)
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

def long_short(data: pd.DataFrame, col: str, k: int = 10):
    '''
    A demo strategy. Long the top 1/k and short the bottom 1/k by equal weights.

    XXX: the bigger the value in the col, the better the stock is.

    NOTE: data should contain DatetimeIndex, permno, stock_exret,
    and a column of prediced value(col), like lgbm (predicted return
    or predicted relevance score).
    '''
    data = data.groupby(level=0, group_keys=False).apply(
        # ascending
        lambda x: x.sort_values(col)
    )
    grouped = data.groupby(level=0, group_keys=False)['stock_exret']
    # monthly stock limit: 50 ~ 100
    f = lambda x: min(max(len(x) // k, 50), 100)
    group_ret_gap = grouped.apply(
        lambda x: x[-f(x):].mean() #- x[:f(x)].mean()
    )
    return group_ret_gap

if __name__ == '__main__':
    data = pd.read_csv('/Users/znw/Code_python/introToFin/output_lgbm.csv')
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    col = 'lgbm'
    port_ret = long_short(data, col).to_frame()

    col = 'stock_exret'
    print('Annualized Sharpe:', sharpe(port_ret, col))
    print('Max 1m loss:', max_1m_loss(port_ret, col))
    print('Max Drawdown:', max_drawdown(port_ret, col))
    strategy_ret = cum_ret(port_ret, col)
    
    # Plot the benchmark return. Benchmark is the equal weighted 
    # portfolio of all the stocks.
    bm_ret = data.groupby(level=0)['stock_exret'].mean().to_frame()
    bm_ret = cum_ret(bm_ret, col)

    plt.plot(strategy_ret)
    plt.plot(bm_ret)
    plt.legend(['strategy', 'benchmark'])
    plt.show()