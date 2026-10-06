#!/bin/bash
set -euo pipefail

# Configuration: edit these values directly; no command-line arguments are needed.
out_name="20250819_test"
# by default leave as /data and /analysis because the directories are specific in sbatch_run_diann.sh
data_dir="/data"
out_dir="/analysis"
fasta_dir="/analysis/fastas"
num_threads=${nthreads:-24}
diann_bin="/diann-2.7.0/diann-linux"
# Optional extra variable modification; leave empty to disable.
#var_mod="UniMod:21,79.966331,STY"
var_mod=""

var_mod_args=(
    --var-mod "UniMod:35,15.994915,M"
    --var-mod "UniMod:1,42.010565,*n"
)
if [[ -n "${var_mod}" ]]; then
    var_mod_args+=(--var-mod "${var_mod}")
fi

#first setup the library
"${diann_bin}" \
    --fasta "${fasta_dir}/human/UP000005640_9606.fasta" \
    --fasta "${fasta_dir}/CHIKV_AF15561/CHIKV_AF15561.fasta" \
    --fasta "${fasta_dir}/cRAP/camprotR_240512_cRAP_20190401_full_tags.fasta" --cont-quant-exclude cRAP- \
    --gen-spec-lib --predictor --fasta-search \
    --threads "${num_threads}" \
    --out-lib "${out_dir}/${out_name}.parquet" \
    --cut "K*,R*" \
    --missed-cleavages 1 \
    --var-mods 2 \
    --met-excision \
    "${var_mod_args[@]}" \
    --fixed-mod "UniMod:4,57.021464,C" \
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
"${diann_bin}" \
    --gen-spec-lib \
    --dir "${data_dir}" \
    --fasta "${fasta_dir}/human/UP000005640_9606.fasta" \
    --fasta "${fasta_dir}/CHIKV_AF15561/CHIKV_AF15561.fasta" \
    --fasta "${fasta_dir}/cRAP/camprotR_240512_cRAP_20190401_full_tags.fasta" --cont-quant-exclude cRAP- \
    --lib "${out_dir}/${out_name}.predicted.speclib" \
    --threads "${num_threads}" \
    --out "${out_dir}/${out_name}.parquet" \
    --cut "K*,R*" \
    --missed-cleavages 1 \
    --var-mods 2 \
    --met-excision \
    "${var_mod_args[@]}" \
    --fixed-mod "UniMod:4,57.021464,C" \
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
