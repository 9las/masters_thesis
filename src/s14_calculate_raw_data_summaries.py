#!/usr/bin/env python

import pandas as pd

# Read data
data = pd.read_csv(filepath_or_buffer = '../data/raw/nettcr_train_swapped_peptide_ls_3_26_peptides_full_tcr_final.csv')

data = (data
 .query('binder == 1'))

data['count_peptide'] = (data
                         .groupby('peptide')['peptide']
                         .transform(func = 'count'))

data = (data
        .groupby(by = ['peptide',
                       'count_peptide',
                       'origin'],
                 as_index = False)
        .agg(count_origin = ('origin',
                             'count'))
        .assign(fraction_origin = lambda x: x['count_origin'] / x['count_peptide']))

data = (data
        .pivot(columns = 'origin',
               index = 'peptide',
               values = 'fraction_origin')
         .fillna(value = 0))
