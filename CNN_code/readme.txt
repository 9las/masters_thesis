This folder contains the basic scripts for training a pan-specific CNN model

### General Scripts ###
The architecture for the model is located in the "CNN_keras_architecture.py" script, and can be loaded into other scripts.

The "keras_utils.py" script contains various functions that may be of use during model training. 
The most important function here is "enc_list_bl_max_len", which is used for applying the embedding to a feature,
and padding the sequence to the maximum length observed for that given feature.

### Model Training Scripts ###
The scripts "parallele_train_model.sh", "train_single_model.sh" and "single_train_keras_cnn_cdr123_pan.py" are all used for training the model
Before running these scripts, make sure to change the "outdir" parameter in "single_train_keras_cnn_cdr123_pan.py" to your own model folder,
as well as the "model_dir" parameter in "paralelle_train_model.sh" and the model directory for "PBS -d {model directory}" in "train_single_model.sh".

If the architecture or keras_utils.py scripts are placed in another directory, make sure to add their path to the "single_train_keras_cnn_cdr123_pan.py"
via the sys.path.append({directory}) function.

The "single_train_keras_cnn_cdr123_pan.py" script contains the python code for training the given model, while the two other scripts
are used for submitting the model training to the queueing system of Computerome.

Once the directories have been changed in the scripts, the model training can be started as follows:
./parallele_train_model.sh
Which submits 20 jobs to Computerome (one for each model)

### Model Prediction Script ###
The model prediction has to be run seperately, since all 20 jobs needs to be finished before we can predict on the test partitions.

Before running the prediction, change the output directory in "predict_keras_cnn_cdr123_pan.py" via the "outdir" parameter.

When this is done, the prediction can be run as follows:
/home/projects/vaccine/people/matjen/master_project/tools/miniconda3/envs/tf/bin/python predict_keras_cnn_cdr123_pan.py

After the prediction is complete, the prediction will be saved under the file name "cv_pred_df.csv"


