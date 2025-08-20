#!/bin/bash
#SBATCH --job-name=diann
#SBATCH --time=12:00:00
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
    --f ${data_path}/raw/1374_1394/RP_2_Nov6_DIA_Slot2-20_1_1374.d \
    --f ${data_path}/raw/1374_1394/RP_17_Nov6_DIA_Slot2-35_1_1394.d \
    --lib ${data_path}/Uniprot_UP000005640_2025_08_11_canonical_human_library-lib.predicted.speclib \
    --fasta ${data_path}/uniprotkb_proteome_UP000005640_AND_revi_2025_08_11.fasta \
    --out ${data_path}/1374_1394_diann_report.tsv \
    --matrices \
    --threads 24 \
    --qvalue 0.01 \
    --mass-acc 15 \
    --mass-acc-ms1 15 \
    --reanalyse \
    --verbose 4
    "