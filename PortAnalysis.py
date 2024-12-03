import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

output = pd.read_csv("output_lgbm.csv", parse_dates=["date"])
model = 'lgbm'

# 画出真实收益率和预测收益率的直方图
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# sns.histplot(output['stock_exret'], kde=True)
# plt.title('True stock excess return')
# plt.subplot(1, 2, 2)
# sns.histplot(output[model], kde=True)
# plt.title('Predicted stock excess return')
# plt.show()

r2 = 1 - np.sum(
        np.square((output['stock_exret'] - output[model]))
    ) / np.sum(np.square(output['stock_exret']))
print(model, ' ', r2)