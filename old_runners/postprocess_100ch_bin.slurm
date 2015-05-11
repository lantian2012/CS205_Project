#!/bin/bash
#SBATCH -J post_100bin # name for job array
#SBATCH -o post_100bin.out #Standard output
#SBATCH -e post_100bin.err #Standard error
#SBATCH -p holyseasgpu  #Partition
#SBATCH -t 4:00:00 #Running time of 30 mins.
#SBATCH --mem 8000 #Memory request
#SBATCH --gres=gpu
#SBATCH -n 1 #Number of cores
#SBATCH -N 1 #All cores on one machine
#SBATCH --mail-type=END      #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=tianlan@g.harvard.edu  #Email to which notifications will be sent

# first arg = job.id


python2.7 post_process.py prediction_100ch_bin.yaml


