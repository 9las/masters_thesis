#!/bin/bash
#PBS -A vaccine -W group_list=vaccine
#PBS -l nodes=1:gpus=1:ppn=4:thinnode,mem=12GB,walltime=1:00:00
#PBS -d /home/projects/vaccine/people/nilsch/masters_thesis
#PBS -e logs/${PBS_JOBNAME}_t${2}v${3}.e${PBS_JOBID}
#PBS -o logs/${PBS_JOBNAME}_t${2}v${3}.o${PBS_JOBID}
###Set the directory above to your own model directory

##$1-Model path
##$2-Test partition
##$3-Validation partition

module load tools
source /home/projects/vaccine/people/nilsch/mambaforge/etc/profile.d/conda.sh
conda activate env

##Run Model
./$1 -t $2 -v $3
