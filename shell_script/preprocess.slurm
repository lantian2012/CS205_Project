#!/bin/bash
#SBATCH -J preprocess # name for job array
#SBATCH -o pre.out #Standard output
#SBATCH -e pre.err #Standard error
#SBATCH -p holyseasgpu #Partition
#SBATCH -t 4:00:00 #Running time of 30 mins.
#SBATCH --mem 1500 #Memory request
#SBATCH -n 1 #Number of cores
#SBATCH -N 1 #All cores on one machine
#SBATCH --mail-type=END      #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=tianlan@g.harvard.edu  #Email to which notifications will be sent

# first arg = job.id

python2.7 prl_preprocess_selected.py $SLURM_ARRAY_TASK_ID 100



