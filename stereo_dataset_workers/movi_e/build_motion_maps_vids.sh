#!/bin/bash

SEQUENCES_ROOT=/Users/ajay/Documents/Visual-Intelligence-Lab/MathWorks_Project/kubric_generated_data/stereo_datasets/movi_e/pure_translation
SCRIPT_DIR=/Users/ajay/Documents/Visual-Intelligence-Lab/MathWorks_Project/Codebases/kubric/post_process_stereo

if [ $# -ne 2 ]; then
    echo "Usage: ./build_motion_maps_vids.sh <start_idx> <end_idx>"
    exit 1
fi

start_idx=$1
end_idx=$2

if ! [[ "$start_idx" =~ ^[0-9]+$ ]] || ! [[ "$end_idx" =~ ^[0-9]+$ ]]; then
    echo "Error: Both arguments must be integers."
    exit 1
fi

if [ "$start_idx" -gt "$end_idx" ]; then
    echo "Error: start_idx ($start_idx) must be <= end_idx ($end_idx)."
    exit 1
fi

cd "$SCRIPT_DIR" || { echo "Error: Cannot cd to $SCRIPT_DIR"; exit 1; }

for i in $(seq "$start_idx" "$end_idx"); do
    seq_dir_path="${SEQUENCES_ROOT}/${i}"
    
    if [ ! -d "$seq_dir_path" ]; then
        echo "Warning: Skipping $seq_dir_path (does not exist)"
        continue
    fi
    
    echo "Processing sequence $i..."
    python build_motion_maps.py --seq_dir "$seq_dir_path" --stitch_video --refine
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed on sequence $i"
        exit 1
    fi
done

echo "Done. Processed sequences $start_idx to $end_idx."