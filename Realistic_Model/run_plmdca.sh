#!/bin/bash

PATH_SCRIPT="plmdca.jl"
MSA_DIR="/home/malbrank/Documents/phylogeny-nico/plot_pf00004-3/msas"
PLMDCA_DIR="/home/malbrank/Documents/phylogeny-nico/plot_pf00004-3/dca"

# Ensure output directory exists
mkdir -p "$PLMDCA_DIR"

# Get all .npy files
FILES=($(find "$MSA_DIR" -maxdepth 1 -type f -name "*.npy"))

# Check if files exist
if [ ${#FILES[@]} -eq 0 ]; then
    echo "No .npy files found in $MSA_DIR"
    exit 1
fi

# Count the total number of files
TOTAL_FILES=${#FILES[@]}
echo "Total files to process: $TOTAL_FILES"

# Run with a max of 64 parallel jobs
parallel -j 64 julia "$PATH_SCRIPT" {} "$PLMDCA_DIR/{/.}_couplings.jld" ::: "${FILES[@]}"

