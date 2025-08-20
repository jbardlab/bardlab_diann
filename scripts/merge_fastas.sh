#!/bin/bash

# Input files
CHIKV_FASTA="/Users/jbard/Library/CloudStorage/SynologyDrive-home/repos/bardlab_diann/libraries/CHIKV_ AF15561/CHIKV_AF15561.fasta"
HUMAN_FASTA="/Users/jbard/Library/CloudStorage/SynologyDrive-home/repos/bardlab_diann/libraries/human/UP000005640_9606.fasta"
crap_FASTA="/Users/jbard/Library/CloudStorage/SynologyDrive-home/repos/bardlab_diann/libraries/cRAP/cRAP_20190401.fasta"

# Output file
OUTPUT_FILE="/Users/jbard/Library/CloudStorage/SynologyDrive-home/repos/bardlab_diann/libraries/human_CHIKV/human_UP000005640_chikv_AF15561_combined.fasta"

# Merge the files
cat "$CHIKV_FASTA" "$HUMAN_FASTA" "$crap_FASTA" > "$OUTPUT_FILE"