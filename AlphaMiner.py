import pandas as pd
import numpy as np
import datetime as dt
from Drawer import Drawer

class AlphaMiner:
    '''
    The estimator of factors.
    The standard of a good factor are:
    IC, ICIR, rng, WinRate, 
    '''
    def __init__(self, data: pd.DataFrame, hyps: dict) -> None:
        '''
        Args:
            111
            data (pd.DataFrame): the result of `Drawer.cal()`.
            hyps (dict): the hyperparameters. Must contain the following keys: `MinCorr`, `MinRng`, `MinWinRate`.
        '''
        self.data = data
        self.mc = hyps['MinCorr']
        self.mr = hyps['MinRng']
        self.mw = hyps['MinWinRate']

    def cal_corr(self):
        '''
        Calculate the correlation between the group index and the annualized compound `stock_exret`.
        '''
        def annualize(col: pd.Series) -> float:
            return (1 + col).prod() ** (12 / len(col)) - 1

    def cal_rng(self):
        pass

    def cal_ir(self):
        pass

    def cal_ic(self):
        pass

if __name__ == '__main__':
    pass