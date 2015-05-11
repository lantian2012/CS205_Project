#!/bin/bash
#SBATCH -J dropout # name for job array
#SBATCH -o dropout.out #Standard output
#SBATCH -e dropout.err #Standard error
#SBATCH -p holyseasgpu  #Partition
#SBATCH -t 4:00:00 #Running time of 30 mins.
#SBATCH --mem 8000 #Memory request
#SBATCH --gres=gpu
#SBATCH -n 1 #Number of cores
#SBATCH -N 1 #All cores on one machine
#SBATCH --mail-type=END      #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=colediamond@g.harvard.edu  #Email to which notifications will be sent

# first arg = job.id


THEANO_FLAGS="device=gpu,nvcc=fastmath,floatX=float32,mode=FAST_RUN" python2.7 runner.py $SLURM_JOB_ID default.yaml dropout.yaml


