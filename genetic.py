import pandas as pd
import numpy as np
from Factory import Factory
from gplearn.genetic import SymbolicTransformer
from gplearn.fitness import make_fitness
from scipy.stats import spearmanr

# 使用gplearn的遗传算法框架挖因子
# 目标是找到在全样本（暂时先这么做）rank_ic均值最大的因子

# NOTE: Set your data path here
data_path = '/Users/znw/Code_python/introToFin_utils/stock_sample.csv'
fac = Factory(data_path)
data = fac.load_data(is_zscore=True)

# 保留一下每个月的行数，求前缀和，用于后面的rank_ic计算
month_num = data.groupby(level=0).size().cumsum().to_list()
month_num.insert(0, 0)

def rank_ic(y, y_pred, w):
    '''
    用作gplearn的fitness函数
    '''
    # 使用全局变量month_num，计算每个月的rank_ic，然后求平均输出
    
    # 检查预测值的方差，若方差过小，认为该因子为常数
    variance = np.var(y_pred)
    if variance < 1e-5:  # 可以调整这个阈值来控制惩罚
        return -1000  # 给予较大的负惩罚，使得这种公式很难被选择

    month_rank_ic = []
    months = len(month_num)
    for i in range(1, months):
        y_true = y[month_num[i-1]:month_num[i]]
        y_predict = y_pred[month_num[i-1]:month_num[i]]
        month_rank_ic.append(spearmanr(y_true, y_predict)[0])

    return abs(sum(month_rank_ic) / (months - 1))

target = make_fitness(function=rank_ic, greater_is_better=True)

feature_names = list(set(data.columns) - set(Factory.label_cols))
X = data[feature_names].values
y = data['stock_exret'].values

function_set = ['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'sin', 'cos', 'tan']

gp = SymbolicTransformer(
    population_size=1000,
    hall_of_fame=100,
    n_components=20,
    generations=10,
    tournament_size=30,
    stopping_criteria=0.08,
    function_set=function_set,
    metric=target,
    parsimony_coefficient=0.0005,
    p_crossover=0.85,
    p_subtree_mutation=0.05,
    p_hoist_mutation=0,
    p_point_mutation=0.05,
    p_point_replace=0.05,
    feature_names=feature_names,
    n_jobs=-1,
    verbose=1,
    random_state=0
)

gp.fit(X, y)

# 获取最后一代最佳的符号表达式
est_gp = gp._best_programs
for program in est_gp:
    print(program)
    print(program.raw_fitness_)