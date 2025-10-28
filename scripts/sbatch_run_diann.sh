#!/bin/bash
#SBATCH --job-name=diann
#SBATCH --time=24:00:00
#SBATCH --nodes=1         # max 32 nodes for partition gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --partition=bigmem
#SBATCH --mem=2929G
#SBATCH --output=/scratch/group/jbardlab/jbard/logs/diann/%x.%j.stdout
#SBATCH --error=/scratch/group/jbardlab/jbard/logs/diann/%x.%j.stderr

image_path="/scratch/group/jbardlab/containers/diann_docker:v0.1.sif"
analysis_dir="/scratch/group/jbardlab/jbard/mass_spec/20251016_tune_phos"
analyze_script="${analysis_dir}/tune_data.sh"
data_dir="/scratch/group/jbardlab/jbard/mass_spec/20251016_tune_phos/data"

nthreads=${SLURM_CPUS_PER_TASK}

singularity exec \
    -B "${data_dir}:/data" \
    -B "${analysis_dir}:/analysis" \
    --env nthreads=${nthreads} \
    "${image_path}"  \
    /bin/bash -c "bash ${analyze_script}"