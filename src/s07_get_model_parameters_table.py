#!/usr/bin/env python

import pandas as pd
import os
import re
import s99_project_functions

config_dict = dict()

config_type_tuple = ('model',
                     'tcr',
                     'peptide',
                     'cdr3')

config_re_path_tuple = ('s97_m\d{3}_config.yaml',
                        's97_et\d{2}_config.yaml',
                        's97_ep\d{2}_config.yaml',
                        's97_e3c\d{2}_config.yaml')

for i in range(len(config_type_tuple)):
    config_type = config_type_tuple[i]
    config_re_path = config_re_path_tuple[i]
    config_path_list = [path for path in os.listdir() if re.search(config_re_path, path)]

    config_list = []
    for config_path in config_path_list:
        config = s99_project_functions.load_config(config_name = config_path)
        config_list.append(config['default'])

    config_dict[config_type] = pd.DataFrame(data = config_list)

config_dict['model'] = (config_dict['model']
                        .merge(right = config_dict['tcr'],
                               how = 'left',
                               on = 'embedder_index_tcr')
                        .merge(right = config_dict['peptide'],
                               how = 'left',
                               on = 'embedder_index_peptide')
                        .merge(right = config_dict['cdr3'],
                               how = 'left',
                               on = 'embedder_index_cdr3'))

config_dict['model'][['model_index',
                      'embedder_index_tcr',
                      'embedder_index_peptide',
                      'embedder_index_cdr3']] = (config_dict['model'][['model_index',
                                                                       'embedder_index_tcr',
                                                                       'embedder_index_peptide',
                                                                       'embedder_index_cdr3']]
                                                 .transform(func = lambda x: pd.to_numeric(arg = x.str.lstrip(to_strip = '0'),
                                                                                           downcast = 'integer')))

config_dict['model'] = (config_dict['model']
                        .sort_values(by = ['model_index']))

config_dict['model'].to_csv(path_or_buf = '../results/s07_table__model_parameters.tsv',
                            sep = '\t',
                            index = False)
