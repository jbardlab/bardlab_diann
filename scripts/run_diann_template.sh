#!/bin/bash
data_dir="/scratch/user/jbard/Datasets/Zhou_2024/raw"
fasta_dir="/scratch/user/jbard/Datasets/Zhou_2024/fastas"
out_dir="/scratch/user/jbard/Datasets/Zhou_2024/diann"
out_name="Zhou2024_diann"

# first setup the library
/diann-2.2.0/diann-linux \
    --fasta "${fasta_dir}/UP000005640_9606.fasta" \
    --fasta "${fasta_dir}/CHIKV_AF15561.fasta" \
    --fasta "${fasta_dir}/camprotR_240512_cRAP_20190401_full_tags.fasta" --cont-quant-exclude cRAP- \
    --gen-spec-lib --predictor --fasta-search \
    --threads 24 \
    --out-lib "${out_dir}/${out_name}.parquet" \
    --cut K*,R* \
    --missed-cleavages 1 \
    --var-mods 3 \
    --met-excision \
    --var-mod UniMod:35,15.994915,M \
    --var-mod UniMod:1,42.010565,*n \
    --fixed-mod UniMod:4,57.021464,C \
    --min-pep-len 7 \
    --max-pep-len 30 \
    --min-pr-charge 1 \
    --max-pr-charge 4 \
    --min-fr-mz 200 \
    --max-fr-mz 1800 \
    --min-pr-mz 300 \
    --max-pr-mz 1800 \
    --mass-acc 15 \
    --mass-acc-ms1 15 \
    --matrices \
    --qvalue 0.01 \
    --reanalyse \
    --verbose 4

# then run the analysis
/diann-2.2.0/diann-linux \
    --dir "${data_dir}" \
    --fasta "${fasta_dir}/UP000005640_9606.fasta" \
    --fasta "${fasta_dir}/CHIKV_AF15561.fasta" \
    --fasta "${fasta_dir}/camprotR_240512_cRAP_20190401_full_tags.fasta" --cont-quant-exclude cRAP- \
    --lib "${out_dir}/${out_name}.predicted.speclib" \
    --threads 24 \
    --out "${out_dir}/${out_name}.parquet" \
    --cut K*,R* \
    --missed-cleavages 1 \
    --var-mods 3 \
    --met-excision \
    --var-mod UniMod:35,15.994915,M \
    --var-mod UniMod:1,42.010565,*n \
    --fixed-mod UniMod:4,57.021464,C \
    --min-pep-len 7 \
    --max-pep-len 30 \
    --min-pr-charge 1 \
    --max-pr-charge 4 \
    --min-fr-mz 200 \
    --max-fr-mz 1800 \
    --min-pr-mz 300 \
    --max-pr-mz 1800 \
    --mass-acc 15 \
    --mass-acc-ms1 15 \
    --matrices \
    --qvalue 0.01 \
    --reanalyse \
    --verbose 4