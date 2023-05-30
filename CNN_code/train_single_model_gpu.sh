#!/bin/bash
#PBS -A vaccine -W group_list=vaccine 
#PBS -l nodes=1:gpus=1:ppn=4:thinnode,mem=12GB,walltime=1:00:00
#PBS -d /home/projects/vaccine/people/nilsch/masters_thesis
###Set the directory above to your own model directory

##$1-Model directory
##$2-Model path
##$3-Test partition
##$4-Validation partition

module load tools
source /home/projects/vaccine/people/nilsch/mambaforge/etc/profile.d/conda.sh
conda activate env

cd $1

##Run Model
./$2 -t $3 -v $4
