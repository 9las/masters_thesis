#!/usr/bin/env python

import pandas as pd
import glob
import s99_project_functions

config_dict = dict()

config_type_tuple = ('model',
                     'tcr',
                     'peptide')

config_glob_path_tuple = ('s97_m??_config.yaml',
                          's97_et??_config.yaml',
                          's97_ep??_config.yaml')

for i in range(len(config_type_tuple)):
    config_type = config_type_tuple[i]
    config_glob_path = config_glob_path_tuple[i]
    config_path_list = glob.glob(pathname = config_glob_path)

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
                               on = 'embedder_index_peptide'))

config_dict['model'][['model_index',
                      'embedder_index_tcr',
                      'embedder_index_peptide']] = (config_dict['model'][['model_index',
                                                                          'embedder_index_tcr',
                                                                          'embedder_index_peptide']]
                                                    .transform(func = lambda x: pd.to_numeric(arg = x.str.lstrip(to_strip = '0'),
                                                                                              downcast = 'integer')))

config_dict['model'] = (config_dict['model']
                        .sort_values(by = ['model_index']))

config_dict['model'].to_csv(path_or_buf = '../results/s07_table__model_parameters.tsv',
                            sep = '\t',
                            index = False)
