from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime as dt

class Drawer:
    '''
    Draw the `stock_exret` per month, grouped by a certain factor.

    Bear in mind that in the original data, `stock_exret` is a realized value in the current
    month, and the factor is a value calculated in the previous month. So, the factor is a
    lagging value.

    Args:
        data (pd.DataFrame): must contain 3 columns: `date`, `stock_exret`, and `factor`.
        n (int): the number of groups.
        f_name (str): the column name of the factor.
    '''
    def __init__(self, data: pd.DataFrame, n: int, f_name: str):
        self.data = data
        self.n = n
        self.f_name = f_name
        if f_name not in data.columns:
            raise ValueError(f'Factor "{f_name}" not found in the data.')
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)

    def cal(self):
        '''
        Group the `data` by months, and for each month, sort the rows by the `factor` value 
        ascendingly, and then divide the rows into `n` groups. Then, calculate the mean
        of `stock_exret` for each group, and then calculate the cumulative sum of `stock_exret`.

        `rng` is the difference between the maximum and minimum cumulative sum of `stock_exret`
        for each month.

        `bm` is the benchmark, we calculate it by taking the mean of all `stock_exret` for each 
        month, and then transform it into cumulative sum.
        '''
        self.data['month'] = self.data.index.to_period('M')
        cut = lambda x: pd.qcut(x, self.n, labels=False, duplicates='drop')
        self.data['group'] = self.data.groupby('month')[self.f_name].transform(cut)
        bm = self.data.groupby('month')['stock_exret'].mean().cumsum()
        self.data = self.data.groupby(['month', 'group'])['stock_exret'].mean().unstack().cumsum()
        self.data['rng'] = self.data.max(axis=1) - self.data.min(axis=1)
        self.data['bm'] = bm
        return self.data
    
    def draw(self):
        '''
        Draw the `stock_exret` per month, grouped by `group`.
        '''
        self.data = self.cal()
        self.data.plot()
        plt.ylabel('stock_exret')
        plt.title(self.f_name)
        plt.show()

if __name__ == '__main__':
    import os

    # demo data
    # demo = pd.read_csv('demo.csv')
    # drawer = Drawer(demo, 3, 'market_equity')
    # drawer.draw()

    # Real data
    f_name = 'fnl_gr1a'
    data_path = os.path.dirname(os.getcwd())
    # NOTE: Define your own data path here
    data_path = os.path.join(data_path, 'introToFin_utils', 'data_z.csv')
    data = pd.read_csv(data_path)
    data = data[['date', 'stock_exret', f_name]]
    drawer = Drawer(data, 10, f_name)
    drawer.draw()