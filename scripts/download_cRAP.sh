#!/bin/bash

# Directory to download to
DOWNLOAD_DIR="/Users/jbard/Library/CloudStorage/SynologyDrive-home/repos/bardlab_diann/libraries/cRAP"

# Raw GitHub URL (not the blob URL)
URL="https://github.com/CambridgeCentreForProteomics/camprotR/raw/master/inst/extdata/cRAP_20190401.fasta.gz"

# File names
COMPRESSED_FILE="$DOWNLOAD_DIR/cRAP_20190401.fasta.gz"
EXTRACTED_FILE="$DOWNLOAD_DIR/cRAP_20190401.fasta"

# Check if extracted file already exists
if [ -f "$EXTRACTED_FILE" ]; then
    echo "File already exists: $EXTRACTED_FILE"
    exit 0
fi

# Create directory if it doesn't exist
mkdir -p "$DOWNLOAD_DIR"

# Download the file if it doesn't exist
if [ ! -f "$COMPRESSED_FILE" ]; then
    curl -L -o "$COMPRESSED_FILE" "$URL"
fi

# Extract the file
gunzip "$COMPRESSED_FILE"