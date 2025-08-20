#!/bin/bash
bash download_proteome.sh

library_folder="/Users/jbard/Library/CloudStorage/SynologyDrive-home/repos/bardlab_diann/libraries/"

/diann-2.2.0/diann-linux \
    --fasta "${library_folder}/human/UP000005640_9606.fasta" \
    --fasta "${library_folder}/CHIKV_ AF15561/CHIKV_AF15561.fasta" \
    --fasta "${library_folder}/cRAP/cRAP_20190401.fasta" --cont-quant-exclude cRAP \
    --gen-spec-lib --predictor --fasta-search \
    --threads 24 \
    --out "${library_folder}/human_CHIKV/human_CHIKV_library" \
    --cut K*,R* \
    --missed-cleavages 3 \
    --var-mods 3 \
    --met-excision \
    --var-mod UniMod:35,15.994915,M \
    --var-mod UniMod:1,42.010565,*n \
    --fixed-mod UniMod:4,57.021464,C \
    --min-pep-len 7 \
    --max-pep-len 30 \
    --min-pr-charge 1 \
    --max-pr-charge 4 \
    --min-pr-mz 300 \
    --max-pr-mz 1800 \
    --mass-acc 15 \
    --mass-acc-ms1 15