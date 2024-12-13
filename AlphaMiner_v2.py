'''
Factor Mining and evaluation framework.
'''

import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import PortAnalysis as pa
import os
import time
from tqdm import tqdm
import statsmodels.api as sm

class AlphaMiner:
    '''
    Analyze a single factor.
    '''

    @staticmethod
    def clean(data_path: str, is_zscore: bool = False):
        '''
        0. 读取数据
        1. 索引换成DatetimeIndex
        2. 清洗数据，收益率没有就删，因子没有就填充当月中位数
        3. 按需要计算z分数, 每月扩展窗口, 均值和标准差取窗口内的, 确保不利用未来数据
        '''
        st = time.time()
        trivials = ['year', 'month', 'stock_ticker', 'comp_name']

        data = pd.read_csv(data_path)
        data.drop(columns=trivials, inplace=True)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)

        data.dropna(subset=['stock_exret'], inplace=True)
        data = data.groupby(level=0).transform(
            lambda x: x.fillna(x.median())
        )

        if is_zscore:
            no_z = ['permno', 'stock_exret']
            cols_z = list(set(data.columns) - set(no_z))
            data_processed = pa.zscore(data, cols_z)
            data = pd.concat([data[no_z], data_processed], axis=1)
        
        print('Time for cleaning:', round(time.time() - st, 2), 's\n')

        return data
    
    def __init__(self, data: pd.DataFrame, fac: str) -> None:
        '''
        Load the data and extract the `fac` column.
        NOTE: the columns of `data` should contain DatetimeIndex, permno, stock_exret, fac
        '''
        self.data = data[['permno', 'stock_exret', fac]]
        self.fac = fac

        # output的索引是月，而且不重复
        self.output = pd.DataFrame(index=self.data.index.unique())

    def rank_ic(self):
        '''
        计算所有时间内的横截面rank ic. 
        t-1期所有股票的因子值和t期所有股票超额收益率的spearman秩相关系数
        '''
        monthly_ric = self.data.groupby(level=0).apply(
            lambda x: x['stock_exret'].rank(method='min').corr(x[self.fac].rank(method='min'), 'spearman') 
        )
        self.output['rank_ic'] = monthly_ric

    def ir(self, window: int = None):
        '''
        如果传入window, 计算前window个月的滚动ir。不足前window个月, 计算之前所有月份的ir。
        如果没有传入window, 计算之前所有月份的ir。

        第一个月ir认为是nan, 因为没有标准差。这样设定不影响绘图。
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
        每个月, 将上个月因子值排序, 均分为k组 (用first方法保证均匀)。
        计算分组收益率和极差。
        '''
        grouped = self.data.groupby(level=0)[self.fac]
        ranked = self.data.copy()

        # 对每个股票生成0～k-1的排名标签
        ranked['rank'] = (grouped.rank(method='first') - 1) // (grouped.size() // k + 1)
        ranked['rank'] = ranked['rank'].astype(int).astype(str)

        # 每个月，对每个排名的股票分别计算收益率均值
        # 索引level 0是datetime，把level 1设置成排名
        ranked.set_index('rank', append=True, inplace=True)
        group_ret = ranked.groupby(level=[0, 1])['stock_exret'].mean().unstack()
        group_ret.columns = 'G' + group_ret.columns
        
        # 求累计收益率和极差
        target = [f'G{i}' for i in range(k)]
        group_ret[target] = pa.cum_ret(group_ret, target)
        group_ret['rng'] = abs(group_ret['G0'] - group_ret[f'G{k - 1}'])

        self.output = pd.concat([self.output, group_ret], axis=1)
        # 存储分组数
        self.group_num = k

    def benchmark(self):
        '''
        计算每月市场所有股票收益率的均值.
        '''
        monthly_bm = self.data.groupby(level=0)['stock_exret'].mean().to_frame()
        self.output['bm'] = pa.cum_ret(monthly_bm, 'stock_exret')

    def alpha_beta(self):
        '''
        计算G0 和 G_{k-1}相对市场基准的表现.
        NOTE: 为简化, 暂时使用全数据集, 不能作为策略依据, 仅供参考
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
        绘制分组收益率, 残差, bm和IC, IR
        如果给了`save_path`就保存在这个目录下, 文件名是因子名
        '''
        target = ['bm', 'rng'] + [f'G{i}' for i in range(self.group_num)]
        assert set(target).issubset(set(self.output.columns)), 'Please calculate group return and benchmark.'

        # 计算极端组相对于全市场的alpha, beta
        ab_result = self.alpha_beta()
        a1, b1, ak, bk = tuple([round(x, 4) for x in ab_result])

        palette = sns.color_palette("tab20", 20)
        plt.figure(figsize=(18, 16))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

        # 分组收益率，极差和benchmark
        ax0 = plt.subplot(gs[0])
        self.output[target].plot(ax=ax0, color=palette)
        ax0.set_ylabel('exret')
        ax0.set_xlabel('month')
        ax0.set_title(self.fac)
        text1 = f'alpha_0 = {a1}, beta_0 = {b1}'
        text2 = f'alpha_{self.group_num - 1} = {ak}, beta_{self.group_num - 1} = {bk}'
        ax0.text(0.1, 0.7, text1, transform=ax0.transAxes)
        ax0.text(0.1, 0.6, text2, transform=ax0.transAxes)

        # rank ic，ir（扩展窗口或滑动窗口）
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
    data_path = '/Users/znw/Code_python/introToFin_utils/stock_sample.csv'
    data = AlphaMiner.clean(data_path)

    # 构造新因子
    # (建议不要再把新因子作为一列放进data，在大量构造的时候会引起内存碎片化)
    news = data[['permno', 'stock_exret']].copy()

    # XXX: IR +
    # aliq_at, ebit_bev, ebitda_mev, eqnpo_12m, eqnpo_me, mispricing_mgmt, ni_me, ocf_at, ope_be, sale_bev, sale_me
    # XXX: IR -
    # bidaskhl_21d, chcsho_12m, eqnetis_at, ivol_capm_25d, rmax1_21d, rmax5_21d, rmax5_rvol_21d, rvol_21d

    news['ir_plus'] = pa.zscore(data['ebitda_mev'] + data['mispricing_mgmt'])
    # news['a1'] = (data['ebit_bev'] + data['ebit_sale'] + data['ebitda_mev']) / 3
    # news['a2'] = pa.zscore(data['rmax5_21d']) + (pa.zscore(data['ebit_bev'] + data['ebit_sale'] + data['ebitda_mev'])) / 3
    # news['a3'] = pa.zscore(data['rmax5_21d']) * pa.zscore(data['ebitda_mev'])
    # news['a4'] = (pa.zscore(data['betabab_1260d']) + pa.zscore(data['bidaskhl_21d'])) / 2

    # news['a5'] = pa.zscore(data['eqnetis_at'])
    # news['a13'] = 0.8 * news['a1'] + 0.2 * news['a3']
    # news['a24_chc'] = (news['a2'] + news['a4'] + pa.zscore(data['chcsho_12m'])) / 3
    
    # print(news.describe())
    # news.to_csv('news_list_1213.csv')
    fac = 'ir_plus'
    miner = AlphaMiner(news, fac)
    miner.rank_ic()
    miner.ir()
    miner.ir(24)
    miner.group()
    miner.benchmark()
    miner.draw()

    port_ret = pa.long_short(news, fac, ascending=False, method='long').to_frame()

    col = 'stock_exret'
    print('Annualized Sharpe:', pa.sharpe(port_ret, col))
    print('Max 1m loss:', pa.max_1m_loss(port_ret, col))
    print('Max Drawdown:', pa.max_drawdown(port_ret, col))
    strategy_ret = pa.cum_ret(port_ret, col)
    
    # Plot the benchmark return. Benchmark is the equal weighted 
    # portfolio of all the stocks.
    bm_ret = data.groupby(level=0)['stock_exret'].mean().to_frame()
    bm_ret = pa.cum_ret(bm_ret, col)

    plt.plot(strategy_ret)
    plt.plot(bm_ret)
    plt.legend(['strategy', 'benchmark'])
    plt.show()

    # XXX: 对大量因子画图
    # no_fac = ['permno', 'stock_exret']
    # fac_list = ['a1']
    # for fac in tqdm(fac_list):
    #     miner = AlphaMiner(news, fac)
    #     miner.rank_ic()
    #     miner.ir()
    #     miner.group()
    #     miner.benchmark()
    #     miner.draw(save_path='/Users/znw/Desktop/FactorFigs', is_show=False)
    #     del miner