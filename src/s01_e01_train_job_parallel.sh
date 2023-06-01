#!/bin/bash
#module load tools

#Keep this
model='../src/s01_train.py'
run_script='s01_e01_train_job.sh'

#Submit each of the 20 models to the queue for training
qsub $run_script -F "${model} 0 1"
sleep 0.5
qsub $run_script -F "${model} 0 2"
sleep 0.5
qsub $run_script -F "${model} 0 3"
sleep 0.5
qsub $run_script -F "${model} 0 4"
sleep 0.5

qsub $run_script -F "${model} 1 0"
sleep 0.5
qsub $run_script -F "${model} 1 2"
sleep 0.5
qsub $run_script -F "${model} 1 3"
sleep 0.5
qsub $run_script -F "${model} 1 4"
sleep 0.5

qsub $run_script -F "${model} 2 0"
sleep 0.5
qsub $run_script -F "${model} 2 1"
sleep 0.5
qsub $run_script -F "${model} 2 3"
sleep 0.5
qsub $run_script -F "${model} 2 4"
sleep 0.5

qsub $run_script -F "${model} 3 0"
sleep 0.5
qsub $run_script -F "${model} 3 1"
sleep 0.5
qsub $run_script -F "${model} 3 2"
sleep 0.5
qsub $run_script -F "${model} 3 4"
sleep 0.5

qsub $run_script -F "${model} 4 0"
sleep 0.5
qsub $run_script -F "${model} 4 1"
sleep 0.5
qsub $run_script -F "${model} 4 2"
sleep 0.5
qsub $run_script -F "${model} 4 3"
sleep 0.5
