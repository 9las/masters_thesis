#!/usr/bin/env python

import pandas as pd
import glob
import s99_project_functions

config_file_path_list = glob.glob(pathname = '*_config.yaml')

config_list = []
for config_file_path in config_file_path_list:
    config = s99_project_functions.load_config(config_name = config_file_path)
    config_list.append(config['default'])

model_parameters_df = pd.DataFrame(data = config_list)

model_parameters_df = (model_parameters_df
                       .assign(model_index = lambda x: pd.to_numeric(arg = (x['model_index']
                                                                            .str.lstrip(to_strip = '0')),
                                                                     downcast = 'integer'))
                       .sort_values(by = ['model_index']))

model_parameters_df.to_csv(path_or_buf = '../results/s07_table__model_parameters.tsv',
                           sep = '\t',
                           index = False)
