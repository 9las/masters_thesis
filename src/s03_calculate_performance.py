#!/usr/bin/env python 
"""
@author Nilas Tim Schüsler
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

data = pd.read_csv(filepath_or_buffer = '../out/cv_pred_df.csv')

def ppv(y_true, y_score):
    """Calculate positive predictive value (PPV)"""
    n_total = len(y_score)
    n_top = np.sum(y_true)
    index_top_sorted = np.argsort(y_score)[:n_total-n_top-1:-1]
    n_true_top = y_true[index_top_sorted].sum()
    ppv = n_true_top / n_top
    return ppv


data = data.groupby(['peptide'])
data = data.apply(func = lambda x:  pd.Series(data = [x.binder.sum(),
                                                      roc_auc_score(y_true = x.binder,
                                                                    y_score = x.prediction),
                                                      roc_auc_score(y_true = x.binder,
                                                                    y_score = x.prediction,
                                                                    max_fpr = 0.1),
                                                      ppv(y_true = x.binder.to_numpy(),
                                                          y_score = x.prediction.to_numpy())],
                                              index = ['count_positive',
                                                       'auc',
                                                       'auc01',
                                                       'ppv'],
                                              dtype = object))
data = data.sort_values(by = ['count_positive'],
                        ascending = False)

data.to_csv(path_or_buf = '../out/model_performance.tsv',
            sep = '\t')  
