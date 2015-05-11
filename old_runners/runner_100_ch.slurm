#!/bin/bash
#SBATCH -J 100_ch # name for job array
#SBATCH -o 100_ch.out #Standard output
#SBATCH -e al100_chl.err #Standard error
#SBATCH -p holyseasgpu  #Partition
#SBATCH -t 10:00:00 #Running time of 30 mins.
#SBATCH --mem 8000 #Memory request
#SBATCH --gres=gpu
#SBATCH -n 1 #Number of cores
#SBATCH -N 1 #All cores on one machine
#SBATCH --mail-type=END      #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=colediamond@g.harvard.edu  #Email to which notifications will be sent

# first arg = job.id

THEANO_FLAGS="device=gpu,nvcc=fastmath,floatX=float32,mode=FAST_RUN" python2.7 runner.py $SLURM_JOB_ID default.yaml conv_100_channels.yaml


