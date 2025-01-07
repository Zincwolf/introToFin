import pandas as pd
import numpy as np
from Factory import Factory
from gplearn.genetic import SymbolicTransformer
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from scipy.stats import spearmanr

# NOTE: The code below will find effective factors using 
# ALL the data in the stock_sample.csv file. We just tried
# to show how to find new factors but not build a strategy
# with future data!

# NOTE: Set your data path here
data_path = 'C:\\CODES\\CODE_PYTHON\\stock_sample.csv'
fac = Factory(data_path)
data = fac.load_data(is_zscore=True)

# Keep the number of rows for each month, 
# calculate the prefix sum for later rank_ic calculation
month_num = data.groupby(level=0).size().cumsum().to_list()
month_num.insert(0, 0)

def rank_ic(y, y_pred, w):
    '''
    Calculate rank IC to render it as the fitness function.
    '''
    # Use the global variable month_num to calculate the rank_ic of each month,
    # and then output the average
    global month_num

    # Check the variance of the predicted value. If the variance is too small,
    # the factor is considered a constant.
    variance = np.var(y_pred)
    if variance < 1e-5:
        # negative reward to avoid this kind of formula
        return -1000

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

function_set = ['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'inv', 'abs', 'tan', 'sin', 'cos']

gp = SymbolicTransformer(   
    population_size=2000,
    hall_of_fame=100,
    n_components=20,
    generations=10,
    tournament_size=25,
    stopping_criteria=0.08,
    function_set=function_set,
    metric=target,
    parsimony_coefficient=0.0003,
    p_crossover=0.7,
    p_subtree_mutation=0.19,
    p_hoist_mutation=0.06,
    p_point_mutation=0.05,
    p_point_replace=0.05,
    feature_names=feature_names,
    n_jobs=-1,
    verbose=1,
    random_state=0
)

gp.fit(X, y)

# Get the best symbolic expression of the last generation
est_gp = gp._best_programs
for program in est_gp:
    print(program)
    print(program.raw_fitness_)