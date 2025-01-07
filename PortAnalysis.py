'''
A tool kit for portfolio analysis.
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from typing import Union, List, Literal

def sharpe(
        data: Union[pd.DataFrame, pd.Series], 
        cols: Union[str, List[str], None] = None
    ):
    '''
    Return the annualized sharpe ratio.
    '''
    if isinstance(data, pd.DataFrame):
        if cols is not None:
            data = data[cols]
        else:
            raise ValueError('Please input columns.')
        
    return data.mean() / data.std() * np.sqrt(12)

def zscore(
        data: Union[pd.DataFrame, pd.Series], 
        cols: Union[str, List[str], None] = None
    ):
    '''
    Return the monthly cross-sectional zscore.
    '''
    if isinstance(data, pd.DataFrame):
        if cols is not None:
            data = data[cols]
        else:
            raise ValueError('Please input columns.')
        
    z = lambda x: (x - x.mean()) / x.std()
    data_processed = data.groupby(level=0).transform(z)
    return data_processed

def cum_ret(
        data: Union[pd.DataFrame, pd.Series], 
        cols: Union[str, List[str], None] = None
    ):
    '''
    Return the cumulative return.
    '''
    if isinstance(data, pd.DataFrame):
        if cols is not None:
            data = data[cols]
        else:
            raise ValueError('Please input columns.')
        
    temp = data + 1
    temp = temp.cumprod() - 1
    return temp

def inv_cum_ret(
        data: Union[pd.DataFrame, pd.Series], 
        cols: Union[str, List[str], None] = None
    ):
    '''
    An inverse function of `cum_ret`.
    '''
    if isinstance(data, pd.DataFrame):
        if cols is not None:
            data = data[cols]
        else:
            raise ValueError('Please input columns.')
        
    temp = data + 1
    temp = temp / temp.shift(1) - 1
    
    temp.iloc[0] = data.iloc[0]

    return temp

def max_1m_loss(
        data: Union[pd.DataFrame, pd.Series], 
        cols: Union[str, List[str], None] = None
    ):
    '''
    Return the max 1 month loss.
    '''
    if isinstance(data, pd.DataFrame):
        if cols is not None:
            data = data[cols]
        else:
            raise ValueError('Please input columns.')
        
    return data.min()

def max_drawdown(
        data: Union[pd.DataFrame, pd.Series], 
        cols: Union[str, List[str], None] = None
    ):
    '''
    Return the max drawdown.
    '''
    if isinstance(data, pd.DataFrame):
        if cols is not None:
            data = data[cols]
        else:
            raise ValueError('Please input columns.')
        
    log_ret = np.log(data + 1)
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

def long_short_original(
        data: pd.DataFrame, 
        col: str, 
        k: int = 10, 
        ascending: bool = True,
        method: Literal['long', 'short', 'long_short'] = 'long'
    ):
    '''
    A demo strategy. Long the top 1/k and short the bottom 1/k by equal weights.

    XXX: the bigger the value in the col, the better the stock is.

    NOTE: data should contain DatetimeIndex, permno, stock_exret,
    and a column of prediced value(col), like lgbm (predicted return
    or predicted relevance score).
    '''
    data = data.groupby(level=0, group_keys=False).apply(
        lambda x: x.sort_values(col, ascending=ascending)
    )
    grouped = data.groupby(level=0, group_keys=False)['stock_exret']
    # monthly stock limit: 50 ~ 100
    f = lambda x: min(max(len(x) // k, 50), 100)

    if method == 'long':
        select = lambda x: x[:f(x)].mean()
    elif method == 'short':
        select = lambda x: - x[-f(x):].mean()
    else:
        select = lambda x: x[:f(x)].mean() - x[-f(x):].mean()
    group_ret_gap = grouped.apply(select)
    return group_ret_gap

def long_short(data: pd.DataFrame, col: str, k: int = 80):
    '''
    Weighted long-short strategy. Rank stocks based on the absolute value of predicted returns,
    then assign weights based on the sign of the predicted return (positive for long, negative for short).
    '''
    # Rank stocks based on the absolute value of predicted returns
    data = data.groupby(level=0, group_keys=False).apply(
        lambda x: x.reindex(x[col].abs().sort_values(ascending=False).index)
    )
    
    grouped = data.groupby(level=0, group_keys=False)

    # The weights of long and short positions
    def weighted_mean(x: pd.DataFrame, is_long: bool=True):
        # Select the top k stocks based on the absolute value of predicted returns
        top_k = x.iloc[:k].copy()
        
        # Determine long and short positions based on the sign of the predicted return
        if is_long:
            # Select stocks with positive predicted returns
            long = top_k[top_k[col] > 0].copy()
            return long['stock_exret'].mean() if long.size > 0 else 0
        else:
            # Select stocks with negative predicted returns
            short = top_k[top_k[col] < 0].copy()
            return short['stock_exret'].mean() if short.size > 0 else 0

    # Calculate the returns of long and short positions
    best_group_ret = grouped.apply(lambda x: weighted_mean(x, is_long=True))
    worst_group_ret = grouped.apply(lambda x: weighted_mean(x, is_long=False))
    
    # Portfolio return
    # The number of positive weights is n+, and the number of negative weights is n-.
    # To deleverage, we need to sum up the absolute weights of long and short positions.
    # and the absolute sum is n+ * (1 / n+) + n- * (1 / n-) = 2
    port = (best_group_ret - worst_group_ret) / 2
    port.name = 'stock_exret'
    return port

if __name__ == '__main__':
    # NOTE: RUN one of the models in the repository and get an output file.
    # DEMO: Take LGBM as an example.
    data = pd.read_csv('C:\\CODES\\CODE_PYTHON\\output_lgbm.csv')
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    # NOTE: for mlp and lambdarank, use the following code.
    # col = 'mlp' OR 'lgbm'(for lambdarank)
    # port_ret = long_short_original(data, col, ascending=False)

    col = 'lgbm'
    port_ret = long_short(data, col)

    # col = 'stock_exret'
    print('Annualized Sharpe:', sharpe(port_ret))
    print('Max 1m loss:', max_1m_loss(port_ret))
    print('Max Drawdown:', max_drawdown(port_ret))
    strategy_ret = cum_ret(port_ret)
    
    # Plot the benchmark return. 
    bm_ret = data.groupby(level=0)['stock_exret'].mean()
    bm_ret = cum_ret(bm_ret)

    plt.plot(strategy_ret)
    plt.plot(bm_ret)
    plt.title('Cumulative Return')
    plt.xlabel('month')
    plt.ylabel('stock excess return')
    plt.legend(['strategy', 'benchmark'])
    plt.show()