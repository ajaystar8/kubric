#!/bin/bash
#
# Runs build_motion_maps.py over a range of stereo sequence directories to
# generate per-frame dynamic object masks and dynamic flow maps for both cameras.
#
# For each sequence index in [start_idx, end_idx], the script locates the
# corresponding directory under SEQUENCES_ROOT, then calls build_motion_maps.py
# with --stitch_video and --refine enabled. Segmentation-refined dynamic masks
# and plasma-colormap flow visualisations are saved under each sequence's
# left_camera/ and right_camera/ subdirectories, and side-by-side stereo MP4
# videos are written to the sequence root.
#
# Usage:
#   ./build_motion_maps_vids.sh <start_idx> <end_idx>
#
# Arguments:
#   start_idx   First sequence index to process (inclusive).
#   end_idx     Last sequence index to process (inclusive).
#
# Configuration (edit at top of script):
#   SEQUENCES_ROOT  Root directory containing numbered sequence subdirectories.
#   SCRIPT_DIR      Directory containing build_motion_maps.py.

SEQUENCES_ROOT=./generation/movi_e/pure_translation
SCRIPT_DIR=./post_process_stereo

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