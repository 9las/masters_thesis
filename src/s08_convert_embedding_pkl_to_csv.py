#!/usr/bin/env python

import pandas as pd
import numpy as np

embedding = pd.read_pickle(filepath_or_buffer = '../data/s01_embedding_tcr_hugging_face_facebook_esm2_t36_3B_UR50D.pkl')

embedding = (embedding
             .applymap(func = lambda x: x.astype(dtype = np.half())))

embedding = (embedding
             .melt(var_name = 'cdr',
                   value_name = 'embedding',
                   ignore_index = False)
             .explode(column = 'embedding')
             .assign(amino_acid_position = lambda x: (x
                                                      .groupby('original_index')
                                                      .cumcount()))
             .explode(column = 'embedding'))
             .assign(embedding_value_position = lambda x: (x
                                                           .groupby(['original_index',
                                                                     'amino_acid_position'])
                                                           .cumcount())))
