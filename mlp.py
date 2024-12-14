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
    optimizer = torch.optim.SGD(
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
        test_dl: DataLoader
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
        for features, labels in test_dl:
            outputs = model(features)
            l.append(loss(outputs, labels))
        print(f'Mean Loss on test dataset: {sum(l) / len(l)}')

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
    data_path = 'E:/Collaborate/FinTech/introToFin_utils/stock_sample.csv'
    black_list_path = None
    white_list_path = 'E:/Collaborate/FinTech/white_list2.txt'

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
    k = 10
    batch_size = 128
    learning_rate = 0.01
    epochs = 10
    weight_decay = 0
    num_hiddens1, num_hiddens2 = 16, 10
    dropout1, dropout2 = 0, 0
    device = 'cpu'

    loss = nn.CrossEntropyLoss() if is_y_rank else nn.MSELoss()

    fac = Factory(data_path, k, is_y_rank, white_list=white_list)
    data = fac.load_data(is_zscore=True)
    n_fac = fac.n_fac

    model = nn.Sequential(
        nn.Linear(n_fac, num_hiddens1),
        nn.ReLU(),
        nn.Dropout(dropout1),
        nn.Linear(num_hiddens1, num_hiddens2),
        nn.ReLU(),
        nn.Dropout(dropout2),
        nn.Linear(num_hiddens2, k)
    )
    # model = nn.Sequential(
    #         nn.Linear(n_fac, k)
    # )
    model.apply(init_weights)

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

        model = train(model, train_dl, loss, epochs, learning_rate, weight_decay, device)
        evaluate(model, loss, val_dl)

        labels = test['stock_exret_rank'] if is_y_rank else test['stock_exret']
        no_features = Factory.label_cols + (['stock_exret_rank'] if is_y_rank else [])
        out = test[Factory.label_cols].copy()
        out['mlp'] = predict(model, test.drop(columns=no_features), labels, is_y_rank, device)
        output.append(out)

    output = pd.concat(output)
    output.to_csv('output_mlp.csv')
