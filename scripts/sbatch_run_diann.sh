#!/bin/bash
#SBATCH --job-name=diann
#SBATCH --time=24:00:00
#SBATCH --nodes=1         # max 32 nodes for partition gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=/scratch/user/jbard/sbatch/logs/diann/%x.%j.stdout
#SBATCH --error=/scratch/user/jbard/sbatch/logs/diann/%x.%j.stderr

image_path="/scratch/group/jbardlab/containers/diann_docker:v0.1.sif"
analyze_script="/scratch/user/jbard/repos/bardlab_diann/scripts/run_diann_template.sh"

singularity exec \
    "${image_path}"  \
    /bin/bash -c "bash ${analyze_script}"