#!/usr/bin/env python

import pandas as pd
import numpy as np

embedding = pd.read_pickle(filepath_or_buffer = '../data/s01_embedding_tcr_hugging_face_facebook_esm2_t36_3B_UR50D.pkl')
data = pd.read_csv(filepath_or_buffer = '../data/raw/nettcr_train_swapped_peptide_ls_3_26_peptides_full_tcr_final.csv',
                   usecols = ['original_index'])

bin_count = 30
embedding_min = embedding.applymap(func = lambda x: np.min(x)).min(axis = None)
embedding_max = embedding.applymap(func = lambda x: np.max(x)).max(axis = None)
bin_edges = np.linspace(start = embedding_min,
                        stop = embedding_max,
                        num = bin_count + 1)


embedding = (embedding
             .applymap(lambda x: np.histogram(a = x,
                                              bins = bin_edges)[0]))


data = (data
        .groupby('original_index')
        .agg(count = ('original_index',
                      'count')))

embedding = (embedding
             .merge(right = data,
                    how = 'left',
                    on = 'original_index'))

del data

embedding = (embedding[['a1_encoded',
                        'a2_encoded',
                        'a3_encoded',
                        'b1_encoded',
                        'b2_encoded',
                        'b3_encoded']]
             .multiply(other = embedding['count'],
                       axis = 'index'))

embedding = embedding.sum().sum()
