'''
A tool kit for port analysis.
'''

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def sharpe(data: pd.DataFrame, col: str):
    '''
    Return the annualized sharpe ratio of a return rate column.
    '''
    return data[col].mean() / data[col].std() * np.sqrt(12)

def cum_ret(data: pd.DataFrame, col: str, is_plot: bool = True):
    '''
    Return the cumulative return of a return rate column.
    '''
    temp = data[col] + 1
    temp = temp.cumprod() - 1 

    if is_plot:
        # TODO: 横坐标等绘图细节可能还要调节
        plt.plot(temp)
        plt.show()
    
    # return temp

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
    best_group_ret = grouped.apply(
        lambda x: x[-(len(x) // k):].mean()
    )
    worst_group_ret = grouped.apply(
        lambda x: x[:(len(x) // k)].mean()
    )
    return best_group_ret - worst_group_ret

if __name__ == '__main__':
    data = pd.read_csv('/Users/znw/Code_python/introToFin/output_lgbm.csv')
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    
    col = 'lgbm'
    port_ret = long_short(data, col).to_frame()
    print(port_ret)

    col = 'stock_exret'
    print('Annualized Sharpe:', sharpe(port_ret, col))
    # print('Cummulative return:', cum_ret(port_ret, col))
    print('Max 1m loss:', max_1m_loss(port_ret, col))
    print('Max Drawdown:', max_drawdown(port_ret, col))
    cum_ret(port_ret, col)