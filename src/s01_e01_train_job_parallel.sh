#!/bin/bash

config='s97_e01_config.yaml'
run_script='s01_e01_train_job.sh'

#Submit each of the 20 models to the queue for training
qsub $run_script -F "${config} 0 1"
sleep 0.5
qsub $run_script -F "${config} 0 2"
sleep 0.5
qsub $run_script -F "${config} 0 3"
sleep 0.5
qsub $run_script -F "${config} 0 4"
sleep 0.5

qsub $run_script -F "${config} 1 0"
sleep 0.5
qsub $run_script -F "${config} 1 2"
sleep 0.5
qsub $run_script -F "${config} 1 3"
sleep 0.5
qsub $run_script -F "${config} 1 4"
sleep 0.5

qsub $run_script -F "${config} 2 0"
sleep 0.5
qsub $run_script -F "${config} 2 1"
sleep 0.5
qsub $run_script -F "${config} 2 3"
sleep 0.5
qsub $run_script -F "${config} 2 4"
sleep 0.5

qsub $run_script -F "${config} 3 0"
sleep 0.5
qsub $run_script -F "${config} 3 1"
sleep 0.5
qsub $run_script -F "${config} 3 2"
sleep 0.5
qsub $run_script -F "${config} 3 4"
sleep 0.5

qsub $run_script -F "${config} 4 0"
sleep 0.5
qsub $run_script -F "${config} 4 1"
sleep 0.5
qsub $run_script -F "${config} 4 2"
sleep 0.5
qsub $run_script -F "${config} 4 3"
sleep 0.5
