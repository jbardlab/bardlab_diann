#!/bin/bash
#docker run -it -v $(pwd):/data diann:latest /bin/bash
#cd /diann-2.2.0
#chmod +x ./diann-linux
# /diann-2.2.0/diann-linux --fasta /data/raw/uniprotkb_proteome_UP000005640_AND_revi_2025_08_11.fasta \
#         --gen-spec-lib --predictor --fasta-search \
#         --threads 4 \
#         --out /data/Uniprot_UP000005640_2025_08_11_canonical_human_library.tsv \
#         --cut K*,R* \
#         --missed-cleavages 3 \
#         --var-mods 3 \
#         --met-excision \
#         --var-mod UniMod:35,15.994915,M \
#         --var-mod UniMod:1,42.010565,*n \
#         --fixed-mod UniMod:4,57.021464,C \
#         --min-pep-len 7 \
#         --max-pep-len 30 \
#         --min-pr-charge 1 \
#         --max-pr-charge 4 \
#         --min-pr-mz 300 \
#         --max-pr-mz 1800 \
#         --mass-acc 10 \
#         --mass-acc-ms1 10

/diann-2.2.0/diann-linux \
    --f /data/raw/RP_2_Nov6_DIA_Slot2-20_1_1374.d \
    --f /data/raw/RP_17_Nov6_DIA_Slot2-35_1_1394.d \
    --lib /data/Uniprot_UP000005640_2025_08_11_canonical_human_library-lib.predicted.speclib \
    --fasta /data/raw/uniprotkb_proteome_UP000005640_AND_revi_2025_08_11.fasta \
    --out /data/1374_1394_diann_report.tsv \
    --threads 4 \
    --qvalue 0.01 \
    --matrices \
    --reanalyse