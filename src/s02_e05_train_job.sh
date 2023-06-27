#!/bin/bash
#PBS -A vaccine -W group_list=vaccine
#PBS -l nodes=1:ppn=4:thinnode,mem=24GB,walltime=3:00:00
#PBS -d /home/projects/vaccine/people/nilsch/masters_thesis/src/
#PBS -e /home/projects/vaccine/people/nilsch/masters_thesis/logs/
#PBS -o /home/projects/vaccine/people/nilsch/masters_thesis/logs/

# $1-Config file path
# $2-Test partition index
# $3-Validation partition index

# Go to working directory
cd $PBS_O_INITDIR

source /home/projects/vaccine/people/nilsch/mambaforge/etc/profile.d/conda.sh
conda activate env

##Run Model
./s02_train.py -c $1 -t $2 -v $3

# Get status
qstat -f -1 $PBS_JOBID
