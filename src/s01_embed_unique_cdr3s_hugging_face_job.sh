#!/bin/bash
#PBS -A vaccine -W group_list=vaccine
#PBS -l nodes=1:gpus=1:ppn=4,mem=188GB,walltime=0:30:00
#PBS -d /home/projects/vaccine/people/nilsch/masters_thesis/src/
#PBS -e /home/projects/vaccine/people/nilsch/masters_thesis/logs/
#PBS -o /home/projects/vaccine/people/nilsch/masters_thesis/logs/
#PBS -m ae -M s123015@student.dtu.dk

# $1-Config file path

# Go to working directory
cd $PBS_O_INITDIR

source /home/projects/vaccine/people/nilsch/mambaforge/etc/profile.d/conda.sh
conda activate env

# Run model
HOME=/home/projects/vaccine/people/nilsch ./s01_embed_unique_cdr3s_hugging_face.py -c $1

# Get status
qstat -f -1 $PBS_JOBID
