#!/bin/bash
#SBATCH -J proprocess # name for job array
#SBATCH -o cat.out #Standard output
#SBATCH -e cat.err #Standard error
#SBATCH -p stats #Partition
#SBATCH -t 1:00:00 #Running time of 30 mins.
#SBATCH --mem 4000 #Memory request
#SBATCH -n 1 #Number of cores
#SBATCH -N 1 #All cores on one machine
#SBATCH --mail-type=END      #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=tianlan@g.harvard.edu  #Email to which notifications will be sent

# first arg = job.id

python prl_pre_process_cat.py


