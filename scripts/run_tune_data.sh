#!/bin/bash
out_name="20250820_alphagranule_phos"
data_dir="/data"
out_dir="/analysis"
fasta_dir="/analysis/fastas"
num_threads=${nthreads:-24}
var_mod="UniMod:21,79.966331,STY"

#first setup the library
/diann-2.2.0/diann-linux \
    --fasta "${fasta_dir}/human/UP000005640_9606.fasta" \
    --fasta "${fasta_dir}/CHIKV_AF15561/CHIKV_AF15561.fasta" \
    --fasta "${fasta_dir}/cRAP/camprotR_240512_cRAP_20190401_full_tags.fasta" --cont-quant-exclude cRAP- \
    --gen-spec-lib --predictor --fasta-search \
    --threads ${num_threads} \
    --out-lib "${out_dir}/${out_name}.parquet" \
    --cut K*,R* \
    --missed-cleavages 1 \
    --var-mods 2 \
    --met-excision \
    --var-mod UniMod:35,15.994915,M \
    --var-mod UniMod:1,42.010565,*n \
    --var-mod ${var_mod} \
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

# then run the initial analysis to identify modified peptides
/diann-2.2.0/diann-linux \
    --gen-spec-lib \
    --dir "${data_dir}" \
    --fasta "${fasta_dir}/human/UP000005640_9606.fasta" \
    --fasta "${fasta_dir}/CHIKV_AF15561/CHIKV_AF15561.fasta" \
    --fasta "${fasta_dir}/cRAP/camprotR_240512_cRAP_20190401_full_tags.fasta" --cont-quant-exclude cRAP- \
    --lib "${out_dir}/${out_name}.predicted.speclib" \
    --threads ${num_threads} \
    --out "${out_dir}/${out_name}.parquet" \
    --cut K*,R* \
    --missed-cleavages 1 \
    --var-mods 2 \
    --met-excision \
    --var-mod UniMod:35,15.994915,M \
    --var-mod UniMod:1,42.010565,*n \
    --var-mod ${var_mod} \
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
    --verbose 4 \
    --export-quant

#then calculating the tuning parameters
/diann-2.2.0/diann-linux \
    --tune-lib "${out_dir}/${out_name}-lib.parquet" \
    --var-mod ${var_mod} \
    --tune-rt --tune-im

#then remake the library using those parameters
/diann-2.2.0/diann-linux \
    --tokens "${out_dir}/${out_name}-lib.dict.txt" \
    --rt-model "${out_dir}/${out_name}-lib.tuned_rt.pt" \
    --im-model "${out_dir}/${out_name}-lib.tuned_im.pt" \
    --fasta "${fasta_dir}/human/UP000005640_9606.fasta" \
    --fasta "${fasta_dir}/CHIKV_AF15561/CHIKV_AF15561.fasta" \
    --fasta "${fasta_dir}/cRAP/camprotR_240512_cRAP_20190401_full_tags.fasta" --cont-quant-exclude cRAP- \
    --gen-spec-lib --predictor --fasta-search \
    --threads ${num_threads} \
    --out-lib "${out_dir}/${out_name}_tuned.speclib" \
    --cut K*,R* \
    --missed-cleavages 1 \
    --var-mods 2 \
    --met-excision \
    --var-mod UniMod:35,15.994915,M \
    --var-mod UniMod:1,42.010565,*n \
    --var-mod ${var_mod} \
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

# finally, re-run the analysis with the tuned library
/diann-2.2.0/diann-linux \
    --dir "${data_dir}" \
    --fasta "${fasta_dir}/human/UP000005640_9606.fasta" \
    --fasta "${fasta_dir}/CHIKV_AF15561/CHIKV_AF15561.fasta" \
    --fasta "${fasta_dir}/cRAP/camprotR_240512_cRAP_20190401_full_tags.fasta" --cont-quant-exclude cRAP- \
    --lib "${out_dir}/${out_name}_tuned.predicted.speclib" \
    --threads ${num_threads} \
    --out "${out_dir}/${out_name}_tuned.parquet" \
    --cut K*,R* \
    --missed-cleavages 1 \
    --var-mods 2 \
    --met-excision \
    --var-mod UniMod:35,15.994915,M \
    --var-mod UniMod:1,42.010565,*n \
    --var-mod ${var_mod} \
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
    --verbose 4 \
    --export-quant