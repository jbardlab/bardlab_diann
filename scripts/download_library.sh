#!/bin/bash

# Script to download and extract UniProt human proteome data
# Only downloads if not already present in proteome/ subdirectory

set -e  # Exit on any error

# Configuration
URL="https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz"
PROTEOME_DIR="/Users/jbard/Library/CloudStorage/SynologyDrive-home/repos/bardlab_diann/libraries/human"
FILENAME="UP000005640_9606.fasta.gz"
EXTRACTED_FILE="UP000005640_9606.fasta"

# Create proteome directory if it doesn't exist
mkdir -p "$PROTEOME_DIR"

# Check if the extracted file already exists
if [ -f "$PROTEOME_DIR/$EXTRACTED_FILE" ]; then
    echo "✓ File already exists: $PROTEOME_DIR/$EXTRACTED_FILE"
    echo "Skipping download and extraction."
    exit 0
fi

# Check if the compressed file exists but hasn't been extracted
if [ -f "$PROTEOME_DIR/$FILENAME" ]; then
    echo "✓ Compressed file exists, extracting..."
    cd "$PROTEOME_DIR"
    gunzip "$FILENAME"
    echo "✓ Extraction complete: $EXTRACTED_FILE"
    exit 0
fi

# Download the file
echo "Downloading UniProt human proteome..."
echo "URL: $URL"
echo "Destination: $PROTEOME_DIR/$FILENAME"

cd "$PROTEOME_DIR"

# Use curl with progress bar and resume capability
if command -v curl >/dev/null 2>&1; then
    curl -L -C - -o "$FILENAME" "$URL"
elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$FILENAME" "$URL"
else
    echo "Error: Neither curl nor wget is available. Please install one of them."
    exit 1
fi

# Verify the download was successful
if [ ! -f "$FILENAME" ]; then
    echo "Error: Download failed!"
    exit 1
fi

echo "✓ Download complete: $FILENAME"

# Extract the file
echo "Extracting $FILENAME..."
gunzip "$FILENAME"

if [ -f "$EXTRACTED_FILE" ]; then
    echo "✓ Extraction complete: $EXTRACTED_FILE"
    
    # Display file info
    echo ""
    echo "File information:"
    echo "Size: $(ls -lh "$EXTRACTED_FILE" | awk '{print $5}')"
    echo "Number of sequences: $(grep -c '^>' "$EXTRACTED_FILE" 2>/dev/null || echo 'Could not count sequences')"
else
    echo "Error: Extraction failed!"
    exit 1
fi

echo ""
echo "UniProt human proteome is ready in: $(pwd)/$EXTRACTED_FILE"