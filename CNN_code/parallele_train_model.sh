#!/bin/bash
module load tools

#Keep this
model='single_train_keras_cnn_cdr123_pan.py'
run_script='train_single_model.sh'

#Change this
model_dir='/home/projects/vaccine/people/nilsch/masters_thesis/CNN_code' 

#Enter model directory
# cd $model_dir

#Submit each of the 20 models to the queue for training
qsub $run_script -F "${model_dir} ${model} 0 1"
sleep 0.5
qsub $run_script -F "${model_dir} ${model} 0 2"
sleep 0.5
qsub $run_script -F "${model_dir} ${model} 0 3"
sleep 0.5
qsub $run_script -F "${model_dir} ${model} 0 4"
sleep 0.5

qsub $run_script -F "${model_dir} ${model} 1 0"
sleep 0.5
qsub $run_script -F "${model_dir} ${model} 1 2"
sleep 0.5
qsub $run_script -F "${model_dir} ${model} 1 3"
sleep 0.5
qsub $run_script -F "${model_dir} ${model} 1 4"
sleep 0.5

qsub $run_script -F "${model_dir} ${model} 2 0"
sleep 0.5
qsub $run_script -F "${model_dir} ${model} 2 1"
sleep 0.5
qsub $run_script -F "${model_dir} ${model} 2 3"
sleep 0.5
qsub $run_script -F "${model_dir} ${model} 2 4"
sleep 0.5

qsub $run_script -F "${model_dir} ${model} 3 0"
sleep 0.5
qsub $run_script -F "${model_dir} ${model} 3 1"
sleep 0.5
qsub $run_script -F "${model_dir} ${model} 3 2"
sleep 0.5
qsub $run_script -F "${model_dir} ${model} 3 4"
sleep 0.5

qsub $run_script -F "${model_dir} ${model} 4 0"
sleep 0.5
qsub $run_script -F "${model_dir} ${model} 4 1"
sleep 0.5
qsub $run_script -F "${model_dir} ${model} 4 2"
sleep 0.5
qsub $run_script -F "${model_dir} ${model} 4 3"
sleep 0.5
