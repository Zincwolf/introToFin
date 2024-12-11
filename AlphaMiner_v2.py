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
            ex = lambda x: x.expanding(1)
            ex_z = lambda x: (ex(x) - ex(x).mean()) / ex(x).std()
            data_processed = data[cols_z].groupby(level=0).transform(ex_z)
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
        # 存储分组数，调色用
        self.group_num = k

    def benchmark(self):
        '''
        计算每月市场所有股票收益率的均值.
        '''
        monthly_bm = self.data.groupby(level=0)['stock_exret'].mean().to_frame()
        self.output['bm'] = pa.cum_ret(monthly_bm, 'stock_exret')

    def draw(self, save_path: str = None, is_show: bool = True):
        '''
        绘制分组收益率, 残差, bm和IC, IR
        如果给了`save_path`就保存在这个目录下, 文件名是因子名
        '''
        target = ['bm', 'rng'] + [f'G{i}' for i in range(self.group_num)]
        assert set(target).issubset(set(self.output.columns)), 'Please calculate group return and benchmark.'

        palette = sns.color_palette("tab20", self.group_num) + sns.color_palette("Set2", 10)
        # 使用 gridspec 创建更灵活的布局
        plt.figure(figsize=(18, 16))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])  # 第一张子图高度是2，第二张子图高度是1

        # 第一张子图
        ax0 = plt.subplot(gs[0])
        self.output[target].plot(ax=ax0, color=palette)
        ax0.set_ylabel('exret')
        ax0.set_xlabel('month')
        ax0.set_title(self.fac)

        # 第二张子图
        target2 = ['rank_ic', 'ir']
        ax1 = plt.subplot(gs[1])
        if set(target2).issubset(set(self.output.columns)):
            self.output[target2].plot(ax=ax1)
            ax1.set_xlabel('month')
            ax1.set_title('Rank IC and IR')

        if save_path is not None:
            save_path = os.path.join(save_path, f'{self.fac}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if is_show:
            plt.show()

        plt.close()

if __name__ == '__main__':

    # NOTE: Set your data path here
    data_path = '/Users/znw/Code_python/introToFin_utils/stock_sample.csv'
    data = AlphaMiner.clean(data_path)

    # fac = 'ret_9_1'
    no_fac = ['permno', 'stock_exret']
    fac_list = list(set(data.columns) - set(no_fac))
    for fac in tqdm(fac_list):
        miner = AlphaMiner(data, fac)
        miner.rank_ic()
        miner.ir()
        miner.group()
        miner.benchmark()
        miner.draw(save_path='/Users/znw/Desktop/FactorFigs', is_show=False)
        del miner