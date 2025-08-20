#!/bin/bash
lib_path="/scratch/group/jbardlab/mass_spec/human_CHIKV/human_CHIKV_library-lib.predicted.speclib"
data_folder="/scratch/user/jbard/Datasets/Zhou_2024/raw"
out_path="/scratch/user/jbard/Datasets/Zhou_2024/Zhou2024_diann"

/diann-2.2.0/diann-linux \
    --dir "${data_folder}" \
    --lib "${lib_path}" \
    --fasta "${fasta_dir}/human/UP000005640_9606.fasta" \
    --fasta "${fasta_dir}/CHIKV_AF15561/CHIKV_AF15561.fasta" \
    --fasta "${fasta_dir}/cRAP/cRAP_20190401.fasta" --cont-quant-exclude cRAP \
    --out "${out_path}" \
    --matrices \
    --threads 24 \
    --qvalue 0.01 \
    --mass-acc 15 \
    --mass-acc-ms1 15 \
    --reanalyse \
    --verbose 4