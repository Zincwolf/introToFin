import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import datetime as dt
from tqdm import tqdm
import PortAnalysis as pa
from typing import Union
import time
from Factory import Factory
    
def get_mlp(
        n_fac: int, 
        num_hiddens1: int, 
        num_hiddens2: int, 
        dropout1: float, 
        dropout2: float,
        k: int
    ):
    return nn.Sequential(
        nn.Linear(n_fac, num_hiddens1),
        nn.ReLU(),
        nn.Dropout(dropout1),
        nn.Linear(num_hiddens1, num_hiddens2),
        nn.ReLU(),
        nn.Dropout(dropout2),
        nn.Linear(num_hiddens2, k)
    )

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.zero_()

def train(
        model: nn.Module, 
        train_dl: DataLoader, 
        loss: Union[nn.MSELoss, nn.CrossEntropyLoss],
        epochs: int, 
        lr: float,
        wd: float = 0,
        device: str = 'cuda'
    ) -> nn.Module:
    '''
    Train the model.

    Args:
        model: the model to train
        train_dl: the training data loader
        epochs: the number of epochs
        lr: the learning rate
    '''
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=wd
    )
    
    for epoch in range(epochs):
        model.train()
        for features, labels in train_dl:
            optimizer.zero_grad()
            outputs = model(features)#.reshape(labels.shape)
            # outputs.to(device)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {l.item()}')
    return model

def evaluate(
        model: nn.Module, 
        loss: Union[nn.MSELoss, nn.CrossEntropyLoss],
        eval_dl: DataLoader
    ):
    '''
    Evaluate the model. Print the loss.

    Args:
        model: the model to evaluate
        test_dl: the test data loader
    '''
    model.eval()
    with torch.no_grad():
        l = []
        for features, labels in eval_dl:
            outputs = model(features)
            l.append(loss(outputs, labels))
        mean_loss = sum(l) / len(l)
        print(f'Mean Loss on valid dataset: {mean_loss}')
        return mean_loss

def predict(
        model: nn.Module, 
        input: pd.DataFrame,
        labels: pd.DataFrame,
        is_y_rank: bool = False,
        device: str = 'cuda'
    ):
    '''
    Args:
        - input: DatetimeIndex + facs
    '''
    model.eval()
    input = torch.tensor(input.values, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(input).cpu().numpy()
    if is_y_rank:
        out = out.argmax(axis=1)
        labels = labels.values
        accuracy = (labels.astype(out.dtype) == out).mean()
        print('Test Accuracy:', accuracy)
    return out
    
if __name__ == '__main__':

    # NOTE: set working directory
    data_path = '/Users/znw/Code_python/introToFin_utils/stock_sample.csv'
    black_list_path = None
    white_list_path = None #'E:/Collaborate/FinTech/white_list2.txt'

    black_list, white_list = None, None
    if black_list_path is not None:
        with open(black_list_path) as f:
            black_list = f.read().splitlines()
    if white_list_path is not None:
        with open(white_list_path) as f:
            white_list = f.read().splitlines()

    # Hyperparameters
    # NOTE: set them before you run!
    is_y_rank = True
    # Group numbers
    k = 10 if is_y_rank else 1
    batch_size = 128
    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    epochs = 10
    weight_decays = [1e-4, 1e-5, 1e-6]
    num_hiddens1, num_hiddens2 = 128, 64
    dropout1, dropout2 = 0.02, 0.02
    device = 'cpu'

    loss = nn.CrossEntropyLoss() if is_y_rank else nn.MSELoss()

    fac = Factory(data_path, k, is_y_rank, white_list=white_list)
    data = fac.load_data(is_zscore=True)
    n_fac = fac.n_fac

    output = []
    # Using expanding window method
    for i in tqdm(range(2007,2021)):
        # train - 8 years, validate - 2 years, test - 1 year
        train_st = dt.datetime(2000, 1, 1)
        train_et = dt.datetime(i, 12, 31)
        val_st = dt.datetime(i + 1, 1, 1)
        val_et = dt.datetime(i + 2, 12, 31)
        test_st = dt.datetime(i + 3, 1, 1)
        test_et = dt.datetime(i + 3, 12, 31)

        train_dl = fac.load_torch_dataset(data, train_st, train_et, batch_size, True, device)
        val_dl = fac.load_torch_dataset(data, val_st, val_et, batch_size, False, device)
        test = data[test_st:test_et]

        # 通过验证集，逐年优化参数
        best_lr, best_wd, min_loss, best_model = 0, 0, 99.0, None
        for lr in learning_rates:
            for wd in weight_decays:
                model = get_mlp(n_fac, num_hiddens1, num_hiddens2, dropout1, dropout2, k)
                model.apply(init_weights)
                model = train(model, train_dl, loss, epochs, lr, wd, device)
                loss_val = evaluate(model, loss, val_dl)
                if loss_val < min_loss:
                    best_lr, best_wd, min_loss, best_model = lr, wd, loss_val, model

        labels = test['stock_exret_rank'] if is_y_rank else test['stock_exret']
        no_features = Factory.label_cols + (['stock_exret_rank'] if is_y_rank else [])
        out = test[Factory.label_cols].copy()
        out['mlp'] = predict(best_model, test.drop(columns=no_features), labels, is_y_rank, device)
        output.append(out)

    output = pd.concat(output)
    output.to_csv('output_mlp.csv')
