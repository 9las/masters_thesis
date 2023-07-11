#!/usr/bin/env python

import pandas as pd
import glob
import s99_project_functions

config_file_path_list = glob.glob(pathname = '*_config.yaml')

config_list = []
for config_file_path in config_file_path_list:
    config = s99_project_functions.load_config(config_name = config_file_path)
    config_list.append(config['default'])

experiment_parameters_df = pd.DataFrame(data = config_list)

experiment_parameters_df = (experiment_parameters_df
                            .assign(experiment_index = lambda x: pd.to_numeric(arg = (x['experiment_index']
                                                                                      .str.lstrip(to_strip = '0')),
                                                                               downcast = 'integer'))
                            .sort_values(by = ['experiment_index']))

experiment_parameters_df.to_csv(path_or_buf = '../results/s06_table__experiment_parameters.tsv',
                                sep = '\t',
                                index = False)
