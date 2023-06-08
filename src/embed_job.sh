#!/bin/bash
#PBS -A vaccine -W group_list=vaccine
#PBS -l nodes=1:gpus=1:ppn=4,mem=188GB,walltime=99:00:00
#PBS -d /home/projects/vaccine/people/nilsch/masters_thesis/src/
#PBS -e /home/projects/vaccine/people/nilsch/masters_thesis/logs/
#PBS -o /home/projects/vaccine/people/nilsch/masters_thesis/logs/
#PBS -m ae -M s123015@student.dtu.dk

# Go to working directory
cd $PBS_O_INITDIR

source /home/projects/vaccine/people/nilsch/mambaforge/etc/profile.d/conda.sh
conda activate env_bio_embeddings

# Run model
./embed_test.py

# Get status
qstat -f -1 $PBS_JOBID
