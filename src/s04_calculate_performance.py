#!/usr/bin/env python 
"""
@author Nilas Tim Schüsler
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import glob

# Functions
def ppv(y_true, y_score):
    """Calculate positive predictive value (PPV)"""
    n_total = len(y_score)
    n_top = np.sum(y_true)
    index_top_sorted = np.argsort(y_score)[:n_total-n_top-1:-1]
    n_true_top = y_true[index_top_sorted].sum()
    ppv = n_true_top / n_top
    return ppv

# Load data
file_path_iterator = glob.glob(pathname = '../data/s03_e*_predictions.tsv')

df_list = []
for file_path in file_path_iterator:
    df = (pd.read_csv(filepath_or_buffer = file_path,
                       sep = '\t')
            .assign(file_path = file_path))

    df_list.append(df)

data = pd.concat(objs = df_list,
                 ignore_index = True)

# Process data

data = (data
        .assign(experiment_index = lambda x: (x['file_path']
                                              .str.removeprefix(prefix = '../data/s03_e')
                                              .str.removesuffix(suffix = '_predictions.tsv')
                                              .str.lstrip(to_strip = '0')))
        .assign(experiment_index = lambda x: pd.to_numeric(x['experiment_index'])))

# Create table with performance metrics per peptide
data = (data
        .groupby(['experiment_index',
                  'peptide'])
        .apply(func = lambda x:  pd.Series(data = [x['binder'].sum(),
                                                   roc_auc_score(y_true = x['binder'],
                                                                 y_score = x['prediction']),
                                                   roc_auc_score(y_true = x['binder'],
                                                                 y_score = x['prediction'],
                                                                 max_fpr = 0.1),
                                                   ppv(y_true = x['binder'].to_numpy(),
                                                       y_score = x['prediction'].to_numpy())],
                                           index = ['count_positive',
                                                    'auc',
                                                    'auc01',
                                                    'ppv'],
                                           dtype = object))
        .sort_values(by = ['experiment_index',
                           'count_positive'],
                     ascending = [True,
                                  False]))

# Create table with mean peformance metrics
data_summary = (data
                .groupby(['experiment_index'])
                .apply(func = lambda x: pd.Series(data = [('mean',
                                                           'weighted_mean'),
                                                          x['count_positive'].sum(),
                                                          (x['auc'].mean(),
                                                           np.average(a = x['auc'],
                                                                      weights = x['count_positive'])),
                                                          (x['auc01'].mean(),
                                                           np.average(a = x['auc01'],
                                                                      weights = x['count_positive'])),
                                                          (x['ppv'].mean(),
                                                           np.average(a = x['ppv'],
                                                                      weights = x['count_positive']))],
                                                  index = ['statistic',
                                                           'count_positive',
                                                           'auc',
                                                           'auc01',
                                                           'ppv'],
                                                  dtype = object))
                .explode(['statistic',
                          'auc',
                          'auc01',
                          'ppv'])
                .sort_values(by = ['statistic',
                                   'auc01'],
                             ascending = False))

# Write tables to files

data.to_csv(path_or_buf = '../data/s04_performance.tsv',
            sep = '\t')

data_summary.to_csv(path_or_buf = '../data/s04_performance_summary.tsv',
                    sep = '\t')
