import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
import seaborn as sns

class AlphaMiner:
    '''
    **Nov 21, Version 1.0**

    The estimator of factors. Also supports drawing grouped returns.

    Bear in mind that in the original data, `stock_exret` is a realized value in the current
    month, and the factor is a value calculated in the previous month. So, the factor is a
    lagging value.

    The standard of a good factor are:
    IC, ICIR, rng, WinRate (XXX: TO BE COMPLETED, like max_drawdown, etc.)
    '''
    def __init__(self, data: pd.DataFrame, f_name: str, n: int) -> None:
        '''
        Args:
            data (pd.DataFrame): must contain 3 columns: `date`, `stock_exret`, and `factor`.
            f_name (str): the column name of the factor.
            n (int): the number of groups.
        '''
        self.data = data
        self.n = n
        self.f_name = f_name
        if f_name not in data.columns:
            raise ValueError(f'Factor "{f_name}" not found in the data.')
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)
        self.data['month'] = self.data.index.to_period('M')

        # The grouped data is a DataFrame that stores the cum mean of `stock_exret` for each group.
        self.grouped_data = self.data.copy()
        self.grouped = False

        # The output is a one-row DataFrame.
        # IC36, IC12: the mean information coefficient of the factor over the past 36/12 months.
        # ICIR: the mean of IC / the std of IC
        # rng: the difference between the max and min cumsum of `stock_exret` for each month.
        # WinRate: the proportion of months where the top / bottom group outperforms the last month.
        self.output = pd.DataFrame(
            columns=['factor', 'PIC36', 'PIC12', 'PICIR', 'RIC36', 'RIC12', 'RICIR', 'rng', 'WinRate']
        )
        self.output.loc[0, 'factor'] = f_name

    def cal(self):
        '''
        Group the `data` by months, and for each month, sort the rows by the `factor` value 
        ascendingly, and then divide the rows into `n` quantiles (also called groups). Then, 
        calculate the mean of `stock_exret` for each group, and then calculate the cumulative 
        sum of `stock_exret`.

        `rng` is the difference between the `stock_exret` of the top and bottom group for 
        each month.

        `bm` is the benchmark, we calculate it by taking the mean of all `stock_exret` for each 
        month, and then transform it into cumulative sum.
        '''
        if self.grouped: 
            return
        self.grouped = True
        
        cut = lambda x: pd.qcut(x, self.n, labels=False, duplicates='drop')
        self.grouped_data['group'] = self.grouped_data.groupby('month')[self.f_name].transform(cut)
        bm = self.grouped_data.groupby('month')['stock_exret'].mean().cumsum()
        self.grouped_data = self.grouped_data.groupby(['month', 'group'])['stock_exret'].mean().unstack().cumsum()
        self.grouped_data['rng'] = np.abs(self.grouped_data[self.n - 1] - self.grouped_data[0])
        self.grouped_data['bm'] = bm
        # return self.grouped_data

    def cal_pearson_ic(self):
        '''
        Calculate the Pearson Information Coefficient of the factor.
        '''
        monthly_ic = self.data.groupby('month').apply(
            lambda x: x['stock_exret'].corr(x[self.f_name])
        )
        self.output['PIC36'] = monthly_ic.tail(36).mean()
        self.output['PIC12'] = monthly_ic.tail(12).mean()
        self.output['PICIR'] = monthly_ic.mean() / monthly_ic.std()

    def cal_rank_ic(self):
        '''
        Calculate the Rank Information Coefficient of the factor.
        '''
        monthly_ic = self.data.groupby('month').apply(
            lambda x: x['stock_exret'].rank().corr(x[self.f_name].rank())
        )
        self.output['RIC36'] = monthly_ic.tail(36).mean()
        self.output['RIC12'] = monthly_ic.tail(12).mean()
        self.output['RICIR'] = monthly_ic.mean() / monthly_ic.std()

    def cal_rng(self):
        '''
        Calculate the `rng` of the factor.

        NOTE: call this method after calling `cal`.
        '''
        self.output['rng'] = self.grouped_data['rng'].mean()

    def cal_winrate(self):
        '''
        Calculate the WinRate of the factor.

        We assume that if PICIR >= 0, then the factor is a positive factor.

        NOTE: call this method after calling `cal`.
        '''
        if self.output.loc[0,'PICIR'] >= 0:
            temp = self.grouped_data.loc[:, self.n - 1]
        else:
            temp = self.grouped_data.loc[:, 0]
        tot_win = np.sum(
            temp.values[:-1] < temp.values[1:]
        )
        self.output['WinRate'] = tot_win / (len(temp) - 1)

    def draw(self):
        '''
        Draw the `stock_exret` per month, grouped by `group`.
        '''
        self.cal()
        # TODO: Find a better palette
        palette = sns.color_palette("muted", 12)
        self.grouped_data.plot(color=palette)
        plt.ylabel('stock_exret')
        plt.title(self.f_name)
        plt.show()

    def analyze(self) -> pd.DataFrame:
        '''
        Yield the analysis of the factor.
        '''
        self.cal_pearson_ic()
        self.cal_rank_ic()
        self.cal()
        self.cal_winrate()
        self.cal_rng()
        return self.output

if __name__ == '__main__':
    import os

    # Real data
    f_name = 'age'

    # NOTE: Define your own data path here
    data_path = os.path.dirname(os.getcwd())
    data_path = os.path.join(data_path, 'introToFin_utils', 'data_z.csv')

    data = pd.read_csv(data_path)
    data = data[['date', 'stock_exret', f_name]]
    drawer = AlphaMiner(data, f_name, 10)
    drawer.draw()

    print(drawer.analyze())

    del drawer