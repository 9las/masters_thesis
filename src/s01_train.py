#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score
import os
import sys
import numpy as np
import pandas as pd
import s99_project_functions
import s98_models
import matplotlib.pyplot as plt
import seaborn as sns
import random
import argparse

#Input arguments for running the model
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
#Test partition, which is excluded from training
parser.add_argument("-t", "--test_partition")
#Validation partition for early stopping
parser.add_argument("-v", "--validation_partition")
args = parser.parse_args()
config_filename = args.config
t = int(args.test_partition)
v = int(args.validation_partition)

# Load config
config = s99_project_functions.load_config(config_filename)

# Set parameters from config
experiment_index = config['default']['experiment_index']
patience = config['default']['patience'] #Patience for Early Stopping
dropout_rate = config['default']['dropout_rate'] #Dropout Rate 
encoding_scheme_name = config['default']['encoding'] #Encoding scheme for amino acid sequences
EPOCHS = config['default']['epochs'] #Number of epochs in the training
batch_size = config['default']['batch_size'] #Number of elements in each batch
embedding_model_name = config['default']['embedding_model_name']
dense_layer_units = config['default']['dense_layer_units']

if embedding_model_name is None:
    use_embeddings = False
else:
    use_embeddings = True

#Makes the plots look better
sns.set()

# Set random seed
seed = config['default']['seed']
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
tf.random.set_seed(seed) # Tensorflow random seed

### Input/Output ###
# Read in data
data = pd.read_csv(filepath_or_buffer = os.path.join('../data/raw', config['default']['data']),
                   usecols = ['A1',
                              'A2',
                              'A3',
                              'B1',
                              'B2',
                              'B3',
                              'peptide',
                              'binder',
                              'partition',
                              'original_index'])

# Import model
model = getattr(s98_models, config['default']['model'])

# Make dataframe with unique peptides and their encoding
peptide_dict = pd.DataFrame({'count': data['peptide'].value_counts()})
peptide_dict = (peptide_dict
    .assign(peptide_encoded = lambda x: x.index.map(lambda x: s99_project_functions.encode_peptide(sequence = x,
                                                                                                   encoding_scheme_name = encoding_scheme_name)))
    .assign(peptide_encoded = lambda x: [element for element in s99_project_functions.pad_sequences(sequence_array = x['peptide_encoded'].tolist(),
                                                                                                    padding_value = -5)]))

peptides_unique_count = peptide_dict.shape[0]
entries_count = data.shape[0]

# Add sample weights to unique peptides dataframe
if config['default']['sample_weight']:
    peptide_dict = (peptide_dict
        .assign(weight = lambda x: np.log2(entries_count/(x['count']))/np.log2(peptides_unique_count))
        #Normalize, so that loss is comparable
        .assign(weight = lambda x: x['weight']*(entries_count/np.sum(x['weight']*x['count']))))
else:
    peptide_dict = (peptide_dict
        .assign(weight = 1))

# Get max sequence lengths for padding
a1_max = data['A1'].map(len).max()
a2_max = data['A2'].map(len).max()
a3_max = data['A3'].map(len).max()
b1_max = data['B1'].map(len).max()
b2_max = data['B2'].map(len).max()
b3_max = data['B3'].map(len).max()
pep_max = peptide_dict.index.map(len).max()

def get_model_input(df,
                    peptide_dict,
                    use_embeddings = True,
                    encoding_scheme_name = None,
                    a1_max = None,
                    a2_max = None,
                    a3_max = None,
                    b1_max = None,
                    b2_max = None,
                    b3_max = None):
    """Prepares the embedding for the input features to the model"""
    df = df.merge(right = peptide_dict, how = 'left', left_on = 'peptide', right_index = True)
    encoded_pep = np.array([element for element in df['peptide_encoded']])/5
    sample_weights = df['weight'].to_numpy()
    targets = df['binder'].to_numpy()

    if use_embeddings:
        file_embeddings = np.load(file = '../data/embedding.npz')
        encoded_a1 = file_embeddings['a1_padded']
        encoded_a2 = file_embeddings['a2_padded']
        encoded_a3 = file_embeddings['a3_padded']
        encoded_b1 = file_embeddings['b1_padded']
        encoded_b2 = file_embeddings['b2_padded']
        encoded_b3 = file_embeddings['b3_padded']
    else:
        # Encode positive TCR
        encoded_a1 = df.query('binder == 1')['A1'].map(lambda x: s99_project_functions.encode_peptide(sequence = x,
                                                                                                      encoding_scheme_name = encoding_scheme_name))
        encoded_a2 = df.query('binder == 1')['A2'].map(lambda x: s99_project_functions.encode_peptide(sequence = x,
                                                                                                      encoding_scheme_name = encoding_scheme_name))
        encoded_a3 = df.query('binder == 1')['A3'].map(lambda x: s99_project_functions.encode_peptide(sequence = x,
                                                                                                      encoding_scheme_name = encoding_scheme_name))
        encoded_b1 = df.query('binder == 1')['B1'].map(lambda x: s99_project_functions.encode_peptide(sequence = x,
                                                                                                      encoding_scheme_name = encoding_scheme_name))
        encoded_b2 = df.query('binder == 1')['B2'].map(lambda x: s99_project_functions.encode_peptide(sequence = x,
                                                                                                      encoding_scheme_name = encoding_scheme_name))
        encoded_b3 = df.query('binder == 1')['B3'].map(lambda x: s99_project_functions.encode_peptide(sequence = x,
                                                                                                      encoding_scheme_name = encoding_scheme_name))
        # Pad positive TCR
        encoded_a1 = s99_project_functions.pad_sequences(sequence_array = encoded_a1.tolist(),
                                                         length_sequence_max = a1_max,
                                                         padding_value = -5)/5
        encoded_a2 = s99_project_functions.pad_sequences(sequence_array = encoded_a2.tolist(),
                                                         length_sequence_max = a2_max,
                                                         padding_value = -5)/5
        encoded_a3 = s99_project_functions.pad_sequences(sequence_array = encoded_a3.tolist(),
                                                         length_sequence_max = a3_max,
                                                         padding_value = -5)/5
        encoded_b1 = s99_project_functions.pad_sequences(sequence_array = encoded_b1.tolist(),
                                                         length_sequence_max = b1_max,
                                                         padding_value = -5)/5
        encoded_b2 = s99_project_functions.pad_sequences(sequence_array = encoded_b2.tolist(),
                                                         length_sequence_max = b2_max,
                                                         padding_value = -5)/5
        encoded_b3 = s99_project_functions.pad_sequences(sequence_array = encoded_b3.tolist(),
                                                         length_sequence_max = b3_max,
                                                         padding_value = -5)/5


    # Map negative TCR IDs to positive TCR IDs
    positive_binder_original_ids = df.query('binder == 1').index
    dict_original_ids_to_positive_tcr_array_ids = {positive_binder_original_ids[i]: i for i in range(len(positive_binder_original_ids))}
    list_original_positive_tcr_array_ids = [dict_original_ids_to_positive_tcr_array_ids[original_index] for original_index in df['original_index']]

    # Get negative TCRs from the positive
    encoded_a1 = encoded_a1[list_original_positive_tcr_array_ids]
    encoded_a2 = encoded_a2[list_original_positive_tcr_array_ids]
    encoded_a3 = encoded_a3[list_original_positive_tcr_array_ids]
    encoded_b1 = encoded_b1[list_original_positive_tcr_array_ids]
    encoded_b2 = encoded_b2[list_original_positive_tcr_array_ids]
    encoded_b3 = encoded_b3[list_original_positive_tcr_array_ids]

    return {'peptide': encoded_pep,
            'a1': encoded_a1,
            'a2': encoded_a2,
            'a3': encoded_a3,
            'b1': encoded_b1,
            'b2': encoded_b2,
            'b3': encoded_b3,
            'target': targets,
            'weight': sample_weights}



#def make_tf_ds(df, encoding):
#    """Prepares the embedding for the input features to the model"""
#    encoded_pep = s99_project_functions.enc_list_bl_max_len(df.peptide, encoding, pep_max)/5 # Divide by 5 to make numbers closer to the interval -1 to 1 - makes model learn better
#    encoded_a1 = s99_project_functions.enc_list_bl_max_len(df.A1, encoding, a1_max)/5
#    encoded_a2 = s99_project_functions.enc_list_bl_max_len(df.A2, encoding, a2_max)/5
#    encoded_a3 = s99_project_functions.enc_list_bl_max_len(df.A3, encoding, a3_max)/5
#    encoded_b1 = s99_project_functions.enc_list_bl_max_len(df.B1, encoding, b1_max)/5
#    encoded_b2 = s99_project_functions.enc_list_bl_max_len(df.B2, encoding, b2_max)/5
#    encoded_b3 = s99_project_functions.enc_list_bl_max_len(df.B3, encoding, b3_max)/5
#    targets = df.binder.values
#    sample_weights = df.sample_weight
#    tf_ds = [encoded_pep,
#             encoded_a1, encoded_a2, encoded_a3, 
#             encoded_b1, encoded_b2, encoded_b3,
#             targets,
#             sample_weights]
#    return tf_ds

def my_numpy_function(y_true, y_pred):    
    try:
        auc = roc_auc_score(y_true, y_pred, max_fpr = 0.1)
    except ValueError:
        #Exception for when a positive observation is not present in a batch
        auc = np.array([float(0)])
    return auc

#Custom metric for AUC 0.1
def auc_01(y_true, y_pred):
    "Allows Tensorflow to use the function during training"
    auc_01 = tf.numpy_function(my_numpy_function, [y_true, y_pred], tf.float64)
    return auc_01

#Creates the directory to save the model in
if not os.path.exists('../results'):
    os.makedirs('../results')

outfile = open("../results/s01_e{}_validation_t{}v{}.tsv".format(experiment_index,t,v), mode = "w")
print("t", "v", "valid_loss", "best_auc_0.1", "best_epoch", sep = "\t", file = outfile)


# Prepare plotting
fig, ax = plt.subplots(figsize=(15, 10))

#Training data
x_train_df = data[(data.partition!=t)&(data.partition!=v)]
train_tensor = get_model_input(df = x_train_df,
                               peptide_dict = peptide_dict,
                               use_embeddings = use_embeddings,
                               encoding_scheme_name = encoding_scheme_name,
                               a1_max = a1_max,
                               a2_max = a2_max,
                               a3_max = a3_max,
                               b1_max = b1_max,
                               b2_max = b2_max,
                               b3_max = b3_max)
# train_tensor = make_tf_ds(x_train_df, encoding = encoding)
# x_train = train_tensor[0:7] #Inputs
# targets_train = train_tensor[7] #Target (0 or 1)
# weights_train = train_tensor[8] #Sample weight for the loss function

#Validation data - Used for early stopping
x_valid_df = data[(data.partition==v)]
valid_tensor = get_model_input(df = x_valid_df,
                               peptide_dict = peptide_dict,
                               use_embeddings = use_embeddings,
                               encoding_scheme_name = encoding_scheme_name,
                               a1_max = a1_max,
                               a2_max = a2_max,
                               a3_max = a3_max,
                               b1_max = b1_max,
                               b2_max = b2_max,
                               b3_max = b3_max)
# valid_tensor = make_tf_ds(x_valid_df, encoding = encoding)
# x_valid = valid_tensor[0:7] #Inputs
# targets_valid = valid_tensor[7] #Target (0 or 1)
# weights_valid = valid_tensor[8] #Sample weight for the loss function


count_features_peptide = train_tensor['peptide'].shape[2]
count_features_tcr = train_tensor['a1'].shape[2]

#Selection of the model to train
model = model(dropout_rate = dropout_rate,
              seed = seed,
              count_features_peptide = count_features_peptide,
              count_features_tcr = count_features_tcr,
              a1_max = a1_max,
              a2_max = a2_max,
              a3_max = a3_max,
              b1_max = b1_max,
              b2_max = b2_max,
              b3_max = b3_max,
              pep_max = pep_max,
              dense_layer_units = dense_layer_units)

#Creates the directory to save the model in
if not os.path.exists('../checkpoint'):
    os.makedirs('../checkpoint')

#Saves the model at the best epoch (based on validation loss or other metric)
ModelCheckpoint = keras.callbacks.ModelCheckpoint(
        filepath = '../checkpoint/s01_e{}_t{}v{}'.format(experiment_index,t,v),
        monitor = "val_auc_01",
        mode = "max",
        save_best_only = True)

#EarlyStopping function used for stopping model training when the model does not improve
EarlyStopping = keras.callbacks.EarlyStopping(
    monitor = "val_auc_01",
    mode = "max",
    patience = patience)

#Callbacks to include for the model training
callbacks_list = [EarlyStopping,
                  ModelCheckpoint
    ]

#Optimizers, loss functions, and additional metrics to track
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = [auc_01, "AUC"],
              weighted_metrics = [])

#Announce Training
print("Training model with test_partition = {} & validation_partition = {}".format(t,v), end = "\n")

#Model training
history = model.fit(x = {"pep": train_tensor['peptide'],
                         "a1": train_tensor['a1'],
                         "a2": train_tensor['a2'],
                         "a3": train_tensor['a3'],
                         "b1": train_tensor['b1'],
                         "b2": train_tensor['b2'],
                         "b3": train_tensor['b3']},
          y = train_tensor['target'],
          batch_size = batch_size,
          epochs = EPOCHS,
          verbose = 2,
          sample_weight = train_tensor['weight'],
          validation_data = ({"pep": valid_tensor['peptide'],
                              "a1": valid_tensor['a1'],
                              "a2": valid_tensor['a2'],
                              "a3": valid_tensor['a3'],
                              "b1": valid_tensor['b1'],
                              "b2": valid_tensor['b2'],
                              "b3": valid_tensor['b3']},
                              valid_tensor['target'],
                              valid_tensor['weight']),
          validation_batch_size = batch_size,
          shuffle = True,
          callbacks=callbacks_list
          )

#Loss and metrics for each epoch during training
valid_loss = history.history["val_loss"]
train_loss = history.history["loss"]
valid_auc = history.history["val_auc"]

#Saving the model at the last epoch of training
#model.save(outdir+'/checkpoint/'+'t.'+str(t)+'.v.'+str(v)+".h5")

#Plotting the losses
ax.plot(train_loss, label='train')
ax.plot(valid_loss, label='validation')
ax.set_ylabel("Loss")
ax.set_xlabel("Epoch")
ax.legend()
ax.set_title('Model t.'+str(t)+'.v.'+str(v))

#Record metrics at checkpoint
valid_best = valid_loss[np.argmax(valid_auc)+1]
best_epoch = np.argmax(valid_auc)+1
best_auc = np.max(valid_auc)

#Records loss and metrics at saved epoch
print(t, v, valid_best, best_auc, best_epoch, sep = "\t", file = outfile)

#Clears the session for the next model
tf.keras.backend.clear_session()

#Close log file
outfile.close()

#Save training/validation loss plot
plt.tight_layout()
plt.show()
fig.savefig('../results/s01_e{}_learning_curves_t{}v{}.png'.format(experiment_index,t,v), dpi=200)
