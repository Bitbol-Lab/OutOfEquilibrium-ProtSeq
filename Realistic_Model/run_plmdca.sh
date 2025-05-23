#!/bin/bash

PATH_SCRIPT="plmdca.jl"
MSA_DIR="/home/malbrank/Documents/phylogeny-nico/plot_pf00004/msas"
PLMDCA_DIR="/home/malbrank/Documents/phylogeny-nico/plot_pf00004/dca"

# Ensure output directory exists
mkdir -p "$PLMDCA_DIR"

# Get all .npy files in the directory
FILES=("$MSA_DIR"/*.npy)

# Check if files exist
if [ ! -e "${FILES[0]}" ]; then
    echo "No .npy files found in $MSA_DIR"
    exit 1
fi

# Count the total number of files
TOTAL_FILES=${#FILES[@]}
echo "Total files to process: $TOTAL_FILES"

# Define function to process files
process_file() {
    local file="$1"
    local outfile="$PLMDCA_DIR/$(basename "$file" .npy)_couplings.jld"
    echo "Processing: $file -> $outfile"
    julia "$PATH_SCRIPT" "$file" "$outfile"
}

# Export function and variable for GNU Parallel
export -f process_file
export PLMDCA_DIR

# Run with a max of 32 parallel jobs
parallel -j 32 process_file ::: "${FILES[@]}"
