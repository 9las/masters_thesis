#!/usr/bin/env python 
"""
@author Nilas Tim Schüsler
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import s99_project_functions
import argparse
import yaml

#Input arguments
parser = argparse.ArgumentParser() 
parser.add_argument("-c", "--config")
args = parser.parse_args()
config_filename = args.config

# Load config
config = s99_project_functions.load_config(config_filename)

experiment_index = config['default']['experiment_index']

data = pd.read_csv(filepath_or_buffer = '../data/s03_e{}_predictions.tsv'.format(experiment_index),
                   sep = '\t')

def ppv(y_true, y_score):
    """Calculate positive predictive value (PPV)"""
    n_total = len(y_score)
    n_top = np.sum(y_true)
    index_top_sorted = np.argsort(y_score)[:n_total-n_top-1:-1]
    n_true_top = y_true[index_top_sorted].sum()
    ppv = n_true_top / n_top
    return ppv


data = (data
        .groupby(['peptide'])
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
        .sort_values(by = ['count_positive'],
                     ascending = False))

data_summary = pd.DataFrame(data = {'statistic': ['mean',
                                                  'weighted_mean'],
                                    'count_positive': data['count_positive'].sum(),
                                    'auc': [data['auc'].mean(),
                                            np.average(a = data['auc'],
                                                       weights = data['count_positive'])],
                                    'auc01': [data['auc01'].mean(),
                                              np.average(a = data['auc01'],
                                                         weights = data['count_positive'])],
                                    'ppv': [data['ppv'].mean(),
                                            np.average(a = data['ppv'],
                                                       weights = data['count_positive'])]})

data.to_csv(path_or_buf = '../data/s04_e{}_performance.tsv'.format(experiment_index),
            sep = '\t')

data_summary.to_csv(path_or_buf = '../data/s04_e{}_performance_summary.tsv'.format(experiment_index),
                    sep = '\t',
                    index = False)
