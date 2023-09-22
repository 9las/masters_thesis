#!/bin/bash
#PBS -A vaccine -W group_list=vaccine
#PBS -l nodes=1:gpus=1:ppn=4,mem=188GB,walltime=4:00:00
#PBS -t 0-2%3
#PBS -d /home/projects/vaccine/people/nilsch/masters_thesis/src/
#PBS -e /home/projects/vaccine/people/nilsch/masters_thesis/logs/
#PBS -o /home/projects/vaccine/people/nilsch/masters_thesis/logs/
#PBS -m ae -M s123015@student.dtu.dk

# $1-Config file path

# Go to working directory
cd $PBS_O_INITDIR

source /home/projects/vaccine/people/nilsch/mambaforge/etc/profile.d/conda.sh
conda activate env

t_array=(0 1 3)
v_array=(1 2 4)

# Run model
./s02_train.py -c $1 -t ${t_array[$PBS_ARRAYID]} -v ${v_array[$PBS_ARRAYID]}

# Get status
qstat -f -1 $PBS_JOBID
