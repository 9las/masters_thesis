#!/bin/bash
#PBS -A vaccine -W group_list=vaccine
#PBS -l nodes=1:ppn=4:thinnode,mem=12GB,walltime=1:00:00
#PBS -d /home/projects/vaccine/people/nilsch/masters_thesis/logs

###Set the directory above to your own model directory

##$1-Model path
##$2-Test partition
##$3-Validation partition

# Go to working directory
cd $PBS_O_INITDIR

module load tools
source /home/projects/vaccine/people/nilsch/mambaforge/etc/profile.d/conda.sh
conda activate env

##Run Model
./$1 -t $2 -v $3
