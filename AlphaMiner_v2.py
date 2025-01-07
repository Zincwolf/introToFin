'''
Factor mining and evaluation framework.
'''

import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import PortAnalysis as pa
from Factory import Factory
import os
import statsmodels.api as sm
from tqdm import tqdm

class AlphaMiner:
    '''
    Analyze a single factor.
    '''
    def __init__(self, data: pd.DataFrame, fac: str) -> None:
        '''
        Load the data and extract the `fac` column.
        NOTE: the columns of `data` should contain DatetimeIndex, permno, stock_exret, fac
        '''
        self.data = data[['permno', 'stock_exret', fac]]
        self.fac = fac

        self.output = pd.DataFrame(index=self.data.index.unique())

    def rank_ic(self):
        '''
        Calculate cross-sectional rank IC for all time periods.

        Rank IC is the Spearman rank correlation between the factor values 
        at month t-1 and the stock excess returns at month t.
        '''
        monthly_ric = self.data.groupby(level=0).apply(
            lambda x: x['stock_exret'].corr(x[self.fac], 'spearman') 
        )
        self.output['rank_ic'] = monthly_ric

    def ir(self, window: int = None):
        '''
        Calculate the historical IR or rolling IR.

        If `window` is not None, calculate the rolling IR with a window of `window` months.
        Otherwise, calculate the historical IR.

        The first month's IR is set to nan because there is no standard deviation.
        '''
        assert 'rank_ic' in self.output.columns, 'Please calculate rank ic first.'
        
        if window is not None:
            roll = lambda x: x.rolling(window, min_periods=1)
            roll_ir = lambda x: roll(x).mean() / roll(x).std()
            self.output[f'ir_{window}'] = roll_ir(self.output['rank_ic'])
        else:
            expand = lambda x: x.expanding(1)
            expand_ir = lambda x: expand(x).mean() / expand(x).std()
            self.output['ir'] = expand_ir(self.output['rank_ic'])

    def group(self, k: int = 10):
        '''
        Calculate the group return and range.

        For each month, rank the factor values of the previous month and divide them
        into k groups (use the `first` method to ensure uniformity).
        '''
        grouped = self.data.groupby(level=0)[self.fac]
        ranked = self.data.copy()

        # Get ranks from 0 to k-1 for each stock
        ranked['rank'] = (grouped.rank(method='first') - 1) // (grouped.size() // k + 1)
        ranked['rank'] = ranked['rank'].astype(int).astype(str)

        # For each month, calculate the average return of each rank
        # The index level 0 is datetime, set level 1 as rank
        ranked.set_index('rank', append=True, inplace=True)
        group_ret = ranked.groupby(level=[0, 1])['stock_exret'].mean().unstack()
        group_ret.columns = 'G' + group_ret.columns
        
        # Calculate cumulative return and range
        target = [f'G{i}' for i in range(k)]
        group_ret[target] = pa.cum_ret(group_ret, target)
        group_ret['rng'] = abs(group_ret['G0'] - group_ret[f'G{k - 1}'])

        self.output = pd.concat([self.output, group_ret], axis=1)
        # store the number of groups
        self.group_num = k

    def benchmark(self):
        '''
        Calculate the benchmark return. 
        
        The benchmark is the equal weighted portfolio of all the 
        stocks in the market.
        '''
        monthly_bm = self.data.groupby(level=0)['stock_exret'].mean().to_frame()
        self.output['bm'] = pa.cum_ret(monthly_bm, 'stock_exret')

    def alpha_beta(self):
        '''
        Calculate the alpha and beta of the extreme groups relative 
        to the market benchmark.

        NOTE: For simplicity, use the entire dataset. Cannot be 
        used in a strategy, only for reference.
        '''
        target = ['bm', 'G0', f'G{self.group_num - 1}']
        assert set(target).issubset(set(self.output.columns)), 'Please calculate group return and benchmark.'

        g1 = pa.inv_cum_ret(self.output, 'G0')
        gk = pa.inv_cum_ret(self.output, f'G{self.group_num - 1}')
        bm = pa.inv_cum_ret(self.output, 'bm')
        bm = sm.add_constant(bm.values)
        g1, gk = g1.values, gk.values

        model1 = sm.OLS(g1, bm)
        modelk = sm.OLS(gk, bm)

        res1 = model1.fit()
        resk = modelk.fit()

        a1, b1 = res1.params
        ak, bk = resk.params

        return (a1, b1, ak, bk)

    def draw(self, save_path: str = None, is_show: bool = True):
        '''
        Draw the group return, benchmark, rank IC, and IR.

        If `save_path` is not None, save the figure to this directory.
        '''
        target = ['bm', 'rng'] + [f'G{i}' for i in range(self.group_num)]
        assert set(target).issubset(set(self.output.columns)), 'Please calculate group return and benchmark.'

        # Calculate the alpha and beta of the extreme groups relative to the market benchmark
        ab_result = self.alpha_beta()
        a1, b1, ak, bk = tuple([round(x, 4) for x in ab_result])

        palette = sns.color_palette("tab20", 20)
        plt.figure(figsize=(18, 16))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

        # Group return, range and benchmark
        ax0 = plt.subplot(gs[0])
        self.output[target].plot(ax=ax0, color=palette)
        ax0.set_ylabel('exret')
        ax0.set_xlabel('month')
        ax0.set_title(self.fac)
        text1 = f'alpha_0 = {a1}, beta_0 = {b1}'
        text2 = f'alpha_{self.group_num - 1} = {ak}, beta_{self.group_num - 1} = {bk}'
        ax0.text(0.1, 0.7, text1, transform=ax0.transAxes)
        ax0.text(0.1, 0.6, text2, transform=ax0.transAxes)

        # Rank IC and IR
        target2 = ['rank_ic'] + [i for i in self.output.columns if i.startswith('ir')]
        if set(target2).issubset(set(self.output.columns)):
            mean_ic_ir = self.output[target2].mean().to_string()
            ax1 = plt.subplot(gs[1])
            self.output[target2].plot(ax=ax1)
            ax1.axhline(0, color='gray', linestyle='--')
            ax1.set_xlabel('month')
            ax1.set_title('Rank IC and IR')
            ax1.set_ylim(-0.7, 0.7)
            ax1.text(0.1, -0.62, mean_ic_ir, transform=ax0.transAxes, fontsize=8)

        if save_path is not None:
            save_path = os.path.join(save_path, f'{self.fac}.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        
        if is_show:
            plt.show()

        plt.close()

if __name__ == '__main__':

    # NOTE: Set your data path here
    data_path = 'C:\\CODES\\CODE_PYTHON\\stock_sample.csv'
    data = Factory.clean(data_path)

    # The dataframe `news` will store new factors.
    # It's advised not to put new factors back to `data` as a column,
    # because it may cause memory fragmentation when constructing a large number of factors.
    news = data[['permno', 'stock_exret']].copy()

    # DEMO: Use the original factors in the data
    # XXX: IR + factors in original data
    # aliq_at, ebit_bev, ebitda_mev, eqnpo_12m, eqnpo_me, mispricing_mgmt, ni_me, ocf_at, ope_be, sale_bev, sale_me
    # XXX: IR - factors in original data
    # bidaskhl_21d, chcsho_12m, eqnetis_at, ivol_capm_25d, rmax1_21d, rmax5_21d, rmax5_rvol_21d, rvol_21d

    news['ir_plus'] = pa.zscore(data['ebitda_mev'] + data['mispricing_mgmt'])
    fac = 'ir_plus'
    miner = AlphaMiner(news, fac)
    miner.rank_ic()
    miner.ir()
    miner.ir(24)
    miner.group(10)
    miner.benchmark()
    miner.draw()

    # After you close the figure, the following code will construct a simple
    # long-short strategy based on the factor you just analyzed.
    port_ret = pa.long_short_original(news, fac, ascending=False, method='long').to_frame()

    col = 'stock_exret'
    print('Annualized Sharpe:', pa.sharpe(port_ret, col))
    print('Max 1m loss:', pa.max_1m_loss(port_ret, col))
    print('Max Drawdown:', pa.max_drawdown(port_ret, col))
    strategy_ret = pa.cum_ret(port_ret, col)
    
    # Plot the benchmark return.
    bm_ret = data.groupby(level=0)['stock_exret'].mean().to_frame()
    bm_ret = pa.cum_ret(bm_ret, col)

    plt.plot(strategy_ret)
    plt.plot(bm_ret)
    plt.legend(['strategy', 'benchmark'])
    plt.show()

    # XXX: The following code can draw the effectiveness figure of all original factors.
    # no_fac = ['permno', 'stock_exret']
    # fac_list = ['a1']
    # for fac in tqdm(fac_list):
    #     miner = AlphaMiner(news, fac)
    #     miner.rank_ic()
    #     miner.ir()
    #     miner.group()
    #     miner.benchmark()
    #     miner.draw(save_path='/FactorFigs', is_show=False)
    #     del miner