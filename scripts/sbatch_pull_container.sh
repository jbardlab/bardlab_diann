#!/bin/bash

## JOB SPECIFICATIONS
#SBATCH --job-name=singularity_pull  #Set the job name to "singularity_pull"
#SBATCH --time=03:00:00              #Set the wall clock limit to 1hr
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks=1                   #Request 4 task
#SBATCH --mem=30G                    #Request 30GB per node
#SBATCH --output=/scratch/group/jbardlab/jbard/logs/singularity_pull.%j"

# set up environment for download
cd /scratch/group/jbardlab/containers
#source /home/jbard/.bashrc
export SINGULARITY_CACHEDIR=$TMPDIR/.singularity
module load WebProxy

# execute download
singularity pull diann_docker:v0.1.sif docker://ghcr.io/jbardlab/diann_docker:v0.1