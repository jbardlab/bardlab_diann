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
data_path="/scratch/user/jbard/Datasets/Pan_2024"

singularity exec \
    "${image_path}"  \
    /bin/bash -c "/diann-2.2.0/diann-linux \
    --dir ${data_path}/raw \
    --lib ${data_path}/Uniprot_UP000005640_2025_08_11_canonical_human_library-lib.predicted.speclib \
    --fasta ${data_path}/uniprotkb_proteome_UP000005640_AND_revi_2025_08_11.fasta \
    --out ${data_path}/Pan2024_diann_report.tsv \
    --matrices \
    --threads 24 \
    --qvalue 0.01 \
    --mass-acc 15 \
    --mass-acc-ms1 15 \
    --reanalyse \
    --verbose 4
    "