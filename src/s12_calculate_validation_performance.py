#!/usr/bin/env python
"""
@author Nilas Tim Schüsler
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import glob
import s99_project_functions
import re

# Functions
def ppv(y_true, y_score):
    """Calculate positive predictive value (PPV)"""
    n_total = len(y_score)
    n_top = np.sum(y_true)
    index_top_sorted = np.argsort(y_score)[:n_total-n_top-1:-1]
    n_true_top = y_true[index_top_sorted].sum()
    ppv = n_true_top / n_top
    return ppv

def get_normalisation_divisor(model_index):
    config_filename_model = 's97_m{}_config.yaml'.format(model_index)
    config_model = s99_project_functions.load_config(config_filename_model)
    tcr_normalization_divisor = config_model['default']['tcr_normalization_divisor']

    return tcr_normalization_divisor

# Load data
file_path_iterator = glob.glob(pathname = '../data/s11_m??_t?v?_predictions_on_validation.tsv')

df_list = []
for file_path in file_path_iterator:
    result = re.search(pattern = r'../data/s11_m(\d{2})_t(\d)v(\d)_predictions_on_validation.tsv',
                       string = file_path)

    model_index_padded = result.group(1)

    df = (pd.read_csv(filepath_or_buffer = file_path,
                       sep = '\t')
            .assign(model_index = int(model_index_padded.lstrip()),
                    test_partition = int(result.group(2)),
                    validation_partition = int(result.group(3)),
                    tcr_normalization_divisor = get_normalisation_divisor(model_index_padded)))

    df_list.append(df)

data = pd.concat(objs = df_list,
                 ignore_index = True)

# Create table with performance metrics per peptide
data = (data
        .groupby(['model_index',
                  'tcr_normalization_divisor',
                  'test_partition',
                  'validation_partition',
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
        .sort_values(by = ['model_index',
                           'tcr_normalization_divisor',
                           'test_partition',
                           'validation_partition',
                           'count_positive'],
                     ascending = [True,
                                  True,
                                  True,
                                  True,
                                  False]))

# Create table with mean peformance metrics
data_summary = (data
                .groupby(['model_index',
                          'tcr_normalization_divisor',
                          'test_partition',
                          'validation_partition'])
                .agg(func = {'count_positive': 'sum',
                             'auc': 'mean',
                             'auc01': 'mean',
                             'ppv': 'mean'})
                .sort_values(by = ['model_index',
                                   'tcr_normalization_divisor',
                                   'test_partition',
                                   'validation_partition',
                                   'count_positive'],
                     ascending = [True,
                                  True,
                                  True,
                                  True,
                                  False]))

# Write tables to files

data_summary.to_csv(path_or_buf = '../data/s12_validation_performance_summary.tsv',
                    sep = '\t')
