#!/bin/bash
#SBATCH -J preprocess # name for job array
#SBATCH -o all.out #Standard output
#SBATCH -e all.err #Standard error
#SBATCH -p stats #Partition
#SBATCH -t 16:00:00 #Running time of 30 mins.
#SBATCH --mem 1000 #Memory request
#SBATCH -n 1 #Number of cores
#SBATCH -N 1 #All cores on one machine
#SBATCH --mail-type=END      #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=tianlan@g.harvard.edu  #Email to which notifications will be sent

# first arg = job.id

python prl_preprocess.py $SLURM_ARRAY_TASK_ID 50
#python prl_preprocess.py 23 50



