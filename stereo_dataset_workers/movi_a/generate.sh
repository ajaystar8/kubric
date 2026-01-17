#!/bin/bash
set -e # exit on error

######################### SETUP ############################

# Stereo Camera Setup
baseline=0.54  # meters
focal_length=35.0  # mm
sensor_width=32.0  # mm

# Scene Parameters
objects_set="clevr" # "clevr" or "kubasic"
min_num_objects=3
max_num_objects=5

# Video Generation Parameters
frame_rate=12  # frames per second
frame_end=24  # total number of frames

# Rendering Parameters
resolution=512x512 # (512x512 or 256x256) (can be something else, but these are the tested ones)

# General Settings
num_sequences=1 # total number of sequences to generate

##################################################################

# Begin sequence generation

root_out_dir="./kubric_generated_data/stereo_datasets/movi_a"
mkdir -p ${root_out_dir}

start=$(find ${root_out_dir} -mindepth 1 -maxdepth 1 -type d ! -name '.*' | wc -l | xargs)
start=$((${start}+1))
end=$((${start}+${num_sequences}-1))

echo "Generating ${num_sequences} Stereo-MOVi-A sequences (${start} to ${end})..."

for i in $(seq ${start} ${end})
do
    # ensure output directory exists
    out_dir=${root_out_dir}/${i}
    mkdir -p ${out_dir}

    docker run --rm --interactive \
            --user $(id -u):$(id -g)    \
            --volume "$(pwd):/kubric"   \
            kubricdockerhub/kubruntu    \
            /usr/bin/python3 stereo_dataset_workers/movi_a/worker.py \
            --min_num_objects=${min_num_objects} \
            --max_num_objects=${max_num_objects} \
            --frame_end=${frame_end} \
            --frame_rate=${frame_rate} \
            --focal_length=${focal_length} \
            --sensor_width=${sensor_width} \
            --baseline=${baseline} \
            --job-dir ${out_dir} \
            --resolution=${resolution} \
            --save_state

done

echo "All ${num_sequences} sequences generated in ${root_out_dir}."
echo "Done."