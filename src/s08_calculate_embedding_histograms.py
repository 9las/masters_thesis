#!/usr/bin/env python

import pandas as pd
import numpy as np
import glob
import s99_project_functions
import os

config_main = s99_project_functions.load_config('s97_main_config.yaml')

data_filename = config_main['default']['data_filename']

data = pd.read_csv(filepath_or_buffer = os.path.join('../data/raw',
                                                     data_filename),
                   usecols = ['A1',
                              'A2',
                              'A3',
                              'B1',
                              'B2',
                              'B3',
                              'peptide',
                              'binder',
                              'original_index'])

orginal_index_counts_df = (data
                           .groupby('original_index')
                           .agg(count = ('original_index',
                                         'count')))
histogram_df = pd.DataFrame()

# Parameters
embedder_index_tcr_max = 5
embedder_index_peptide_max = 2
embedder_index_tcr_tuple = tuple(range(1,
                                       embedder_index_tcr_max + 1))
embedder_index_peptide_tuple = tuple(range(1,
                                           embedder_index_peptide_max + 1))
tcr_bin_count_tuple = (20, 30, 30, 30, 30)
peptide_bin_count_tuple = (20, 30)

# Calculate histograms for CDRs
for i in range(len(embedder_index_tcr_tuple)):
    embedder_index_tcr = embedder_index_tcr_tuple[i]
    embedder_index_tcr_padded = str(embedder_index_tcr).zfill(2)
    tcr_bin_count = tcr_bin_count_tuple[i]
    config_tcr = s99_project_functions.load_config('s97_et{}_config.yaml'.format(embedder_index_tcr_padded))
    embedder_name_tcr = config_tcr['default']['embedder_name_tcr']
    embedder_source_tcr = config_tcr['default']['embedder_source_tcr']

    if embedder_source_tcr == 'in-house':
        embedding = (s99_project_functions
                     .encode_unique_tcrs(df = data,
                                         encoding_name = embedder_name_tcr))
    else:
        embedding = pd.read_pickle(filepath_or_buffer = '../data/s01_et{}_embedding.pkl'.format(embedder_index_tcr_padded))

    embedding_min = embedding.applymap(func = lambda x: np.min(x)).min(axis = None)
    embedding_max = embedding.applymap(func = lambda x: np.max(x)).max(axis = None)
    bin_edges = np.linspace(start = embedding_min,
                            stop = embedding_max,
                            num = tcr_bin_count + 1)


    embedding = (embedding
                 .applymap(lambda x: np.histogram(a = x,
                                                  bins = bin_edges)[0])
                 .merge(right = orginal_index_counts_df,
                        how = 'left',
                        on = 'original_index'))

    embedding = (embedding[['a1_encoded',
                            'a2_encoded',
                            'a3_encoded',
                            'b1_encoded',
                            'b2_encoded',
                            'b3_encoded']]
                 .multiply(other = embedding['count'],
                           axis = 'index')
                 .sum()
                 .sum())

    histogram_df = pd.concat([histogram_df,
                              pd.DataFrame({'sequence_type': 'cdr',
                                            'embedder_index': embedder_index_tcr,
                                            'embedder_source': embedder_source_tcr,
                                            'embedder_name': embedder_name_tcr,
                                            'bin_edge': bin_edges[:-1],
                                            'count': embedding})])

# Calculate histograms for peptides
for i in range(len(embedder_index_peptide_tuple)):
    embedder_index_peptide = embedder_index_peptide_tuple[i]
    embedder_index_peptide_padded = str(embedder_index_tcr).zfill(2)
    peptide_bin_count = peptide_bin_count_tuple[i]
    config_peptide = s99_project_functions.load_config('s97_ep{}_config.yaml'.format(embedder_index_peptide_padded))
    embedder_name_peptide = config_peptide['default']['embedder_name_peptide']
    embedder_source_peptide = config_peptide['default']['embedder_source_peptide']

    if embedder_source_tcr == 'in-house':
        embedding = (s99_project_functions
                     .encode_unique_peptides(df = data,
                                             encoding_name = embedder_name_peptide))
    else:
        embedding = pd.read_pickle(filepath_or_buffer = '../data/s01_ep{}_embedding.pkl'.format(embedder_index_peptide_padded))

    embedding_min = embedding['peptide_encoded'].apply(func = lambda x: np.min(x)).min()
    embedding_max = embedding['peptide_encoded'].apply(func = lambda x: np.max(x)).max()
    bin_edges = np.linspace(start = embedding_min,
                            stop = embedding_max,
                            num = tcr_bin_count + 1)

    embedding = (embedding
                 .assign(peptide_encoded = lambda x: (x['peptide_encoded']
                                                      .apply(func = lambda y: np.histogram(a = y,
                                                                                           bins = bin_edges)[0]))))

    embedding = (embedding['peptide_encoded']
                 .multiply(other = embedding['count'],
                           axis = 'index')
                 .sum())

    histogram_df = pd.concat([histogram_df,
                              pd.DataFrame({'sequence_type': 'peptide',
                                            'embedder_index': embedder_index_peptide,
                                            'embedder_source': embedder_source_peptide,
                                            'embedder_name': embedder_name_peptide,
                                            'bin_edge': bin_edges[:-1],
                                            'count': embedding})])

histogram_df = (histogram_df
                .assign(fraction = lambda x: (x
                                              .groupby(['sequence_type',
                                                        'embedder_index'])['count']
                                              .transform(func = lambda y: y / y.sum()))))

histogram_df.to_csv(path_or_buf = '../results/s08_embedding_histograms.tsv',
                    sep = '\t')
