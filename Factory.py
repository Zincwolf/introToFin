import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import datetime as dt
from tqdm import tqdm
from typing import Union
import time
import random
from sklearn.model_selection import train_test_split

class Factory:
    '''
    Load data, pre-process data, generate dataset.

    NOTE: No DataFrame is stored in a Factory instance.
    So you should run `load_data` to get the cleaned data,
    and run `load_torch_dataset` or `load_lgbm_dataset` to
    get the dataset.
    '''

    label_cols = ['permno', 'stock_exret']

    def __init__(
            self,
            data_path: str,  
            k: int = 10, 
            is_y_rank: bool = False,
            black_list: list = None,
            white_list: list = None
        ):
        '''
        Initialize the data processing requirements.

        ## Args:
            - data_path (str): the path of your raw data.
            - k (int, default 10): divide the stocks in each month to k groups by their return ranks.
            - is_y_rank (bool, default False): if `True`, the label of datasets will be return ranks.
            - black_list (list, default None): contains factors that will be dropped from the raw data.
            - white_list (list, default None): the factors remaining in the data.
        '''
        self.k = k
        self.is_y_rank = is_y_rank
        self.data_path = data_path
        self.black_list = black_list
        self.white_list = white_list

    @staticmethod
    def zscore(
            data: Union[pd.DataFrame, pd.Series], 
            cols: Union[str, list[str], None] = None
        ):
        '''
        Return the monthly cross-sectional zscore of certain columns in `data`.
        '''
        if isinstance(data, pd.DataFrame):
            if cols is not None:
                data = data[cols]
            else:
                raise ValueError('Please input columns.')
            
        z = lambda x: (x - x.mean()) / x.std()
        data_processed = data.groupby(level=0).transform(z)
        return data_processed

    @staticmethod
    def clean(data_path: str, is_zscore: bool = False):
        '''
        0. Read data from `data_path`.
        1. Set index to `pd.DatetimeIndex` and drop the trivial 
        columns: year, month, stock_ticker and comp_name.
        2. Clean the data. Drop rows where stock_exret is missing. 
        Fill the missing factor values with the median at current month.
        3. Calculate z-scores for all the factors if `is_zscore=True`.
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
            no_z = Factory.label_cols.copy()
            cols_z = list(set(data.columns) - set(no_z))
            data_processed = Factory.zscore(data, cols_z)
            data = pd.concat([data[no_z], data_processed], axis=1)

        print('Time for cleaning:', round(time.time() - st, 2), 's')
        print('Cleaned Data Shape:', data.shape)

        return data

    def load_data(self, is_zscore: bool = False) -> pd.DataFrame:
        '''
        Load data for dataset creation.
        '''

        data = Factory.clean(self.data_path, is_zscore)

        if self.white_list is not None:
            data = data[Factory.label_cols + self.white_list]
        if self.black_list is not None:
            data.drop(columns=self.black_list, inplace=True)

        print('Data Shape - black & white list:', data.shape)

        self.n_fac = len(data.columns) - 2

        # Generate group rank based on stock_exret
        # The higher the stock_exret is, the bigger the group rank
        if self.is_y_rank:
            grouped = data.groupby(level=0)['stock_exret']
            data['stock_exret_rank'] = (grouped.rank(method='min') - 1) // (grouped.size() // self.k + 1)

        return data
    
    def load_torch_dataset(
            self, 
            data: pd.DataFrame,
            st: dt.datetime, 
            et: dt.datetime,
            batch_size: int,
            is_train: bool = True,
            device: str = 'cuda'
        ) -> DataLoader:
        '''
        Create PyTorch DataLoader with given data.

        ## Args:
            - data (pd.DataFrame): contain features and labels.
            - st (dt.datetime): the start datetime of your dataset. 
            - et (dt.datetime): the end datetime of your dataset.
            - batch_size (int): the size of a minibatch.
            - is_train (bool, default True): if true, shuffle data before creating DataLoader.
            - device (str, default 'cuda'): Try 'mps' for MacBook. 
        '''
        no_feature = Factory.label_cols.copy()
        label_col = 'stock_exret'
        if self.is_y_rank:
            label_col = 'stock_exret_rank'
            no_feature += [label_col]
        features = data.loc[st:et, :].drop(columns=no_feature).values
        labels = data.loc[st:et, label_col].values

        features = torch.tensor(features, dtype=torch.float32, device=device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
                
        dataset = TensorDataset(features, labels)
        return DataLoader(dataset, batch_size, is_train)

    def load_lgbm_dataset_original(
            self,
            data: pd.DataFrame,
            val_st: dt.datetime,
            val_year: int = 2
        ):
        '''
        Split the data into train and valid sets. 
        Data are shuffled in train set and valid set respectively.

        ## Args:
            - val_st (datetime): the start time (month) of valid set. 
            - val_year (int, default = 2): the number of years in the valid set.
        '''
        def shuffle_data(data: pd.DataFrame):
            # Shuffle the index for random batches.
            # The DatetimeIndex is not unique so we should shuffle on integer index
            data.reset_index(inplace=True)
            rand_idx = data.index.to_list()
            random.shuffle(rand_idx)
            data = data.reindex(pd.Index(rand_idx))
            data.set_index('date', inplace=True)
            return data
        
        val_et = val_st + pd.DateOffset(years=val_year, days=-1)
        val = shuffle_data(data[val_st:val_et])
        train = shuffle_data(data[:val_st - pd.DateOffset(months=1)])

        no_feature = Factory.label_cols.copy()
        label_col = 'stock_exret'
        if self.is_y_rank:
            label_col = 'stock_exret_rank'
            no_feature += [label_col]
        
        X_train = train.drop(columns=no_feature)
        X_val = val.drop(columns=no_feature)
        y_train, y_val = train[label_col], val[label_col]

        return X_train, X_val, y_train, y_val

    def split_data(
        self, 
        data: pd.DataFrame,
        val_st: dt.datetime
    ):
        '''
        Split the data into train and validation sets.
        
        ## Args:
            - data (pd.DataFrame): Stock data.
            - val_st (datetime): Start date of validation set (2 years duration).
        
        ## Returns:
            - tuple: (X_train, X_val, y_train, y_val)
        '''
        val_et = val_st + pd.DateOffset(years=2) - dt.timedelta(days=1)
        train_st = dt.datetime(2000,1,1)
        len_val = len(data.loc[val_st:val_et])
        
        # Select features and labels
        # feature selection proves to be useless here

        features = data.loc[train_st:val_et].drop(columns=['stock_exret','permno'])
        labels = data.loc[train_st:val_et, 'stock_exret']

        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=len_val, shuffle=False
        )
        return X_train, X_val, y_train, y_val