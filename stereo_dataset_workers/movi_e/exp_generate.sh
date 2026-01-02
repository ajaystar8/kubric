#!/bin/bash
set -e # exit on error

######################### SETUP ############################

# Stereo Camera Setup
baseline=0.54  # meters
focal_length=35.0  # mm
sensor_width=32.0  # mm
stereo_type="pure_translation" # ("pure_translation" or "lookat_orbit")
max_camera_movement=8.0   # maximum camera movement in meters over the whole sequence

camera_movement="linear_movement" # ("linear_movement" or "linear_movement_linear_lookat") - valid only for "lookat_orbit" stereo type

# Scene Parameters
min_num_static_objects=1
max_num_static_objects=1
min_num_dynamic_objects=1
max_num_dynamic_objects=1

# Video Generation Parameters
frame_rate=2  # frames per second
frame_end=2  # total number of frames

# Rendering Parameters
resolution=512x512 # (512x512 or 256x256) (can be something else, but these are the tested ones)

# General Settings
num_sequences=1 # total number of sequences to generate

##################################################################

# Begin sequence generation

if [ "${stereo_type}" == "lookat_orbit" ]; then
  root_out_dir=./generation/stereo_datasets/movi_e/${stereo_type}/${camera_movement} # root output directory
else
  root_out_dir=./generation/stereo_datasets/movi_e/${stereo_type} # root output directory
fi
mkdir -p ${root_out_dir}

start=$(find ${root_out_dir} -mindepth 1 -maxdepth 1 -type d ! -name '.*' | wc -l | xargs)
start=$((${start}+1))
end=$((${start}+${num_sequences}-1))

if [ "${stereo_type}" == "lookat_orbit" ]; then
  echo "Generating ${num_sequences} Stereo-MOVi-E sequences (stereo_type: ${stereo_type}, camera_movement: ${camera_movement}) (${start} to ${end})..."
else
  echo "Generating ${num_sequences} Stereo-MOVi-E sequences (stereo_type: ${stereo_type}) (${start} to ${end})..."
fi

for i in $(seq ${start} ${end})
do
    # ensure output directory exists
    out_dir=${root_out_dir}/${i}
    mkdir -p ${out_dir}

    if [ "${stereo_type}" == "pure_translation" ]; then
         docker run --rm --interactive \
            --user $(id -u):$(id -g)    \
            --volume "$(pwd):/kubric"   \
            kubricdockerhub/kubruntu    \
            /usr/bin/python3 stereo_dataset_workers/movi_e/pure_translation.py \
            --min_num_static_objects=${min_num_static_objects} \
            --max_num_static_objects=${max_num_static_objects} \
            --min_num_dynamic_objects=${min_num_dynamic_objects} \
            --max_num_dynamic_objects=${max_num_dynamic_objects} \
            --max_camera_movement=${max_camera_movement} \
            --frame_end=${frame_end} \
            --frame_rate=${frame_rate} \
            --focal_length=${focal_length} \
            --sensor_width=${sensor_width} \
            --baseline=${baseline} \
            --job-dir ${out_dir} \
            --resolution=${resolution} \
            --save_state
    else
          docker run --rm --interactive \
            --user $(id -u):$(id -g)    \
            --volume "$(pwd):/kubric"   \
            kubricdockerhub/kubruntu    \
            /usr/bin/python3 stereo_dataset_workers/movi_e/lookat_orbit.py \
            --min_num_static_objects=${min_num_static_objects} \
            --max_num_static_objects=${max_num_static_objects} \
            --min_num_dynamic_objects=${min_num_dynamic_objects} \
            --max_num_dynamic_objects=${max_num_dynamic_objects} \
            --camera=${camera_movement} \
            --max_camera_movement=${max_camera_movement} \
            --frame_end=${frame_end} \
            --frame_rate=${frame_rate} \
            --focal_length=${focal_length} \
            --sensor_width=${sensor_width} \
            --baseline=${baseline} \
            --job-dir ${out_dir} \
            --resolution=${resolution} \
            --save_state
    fi
   
done

echo "All ${num_sequences} sequences generated in ${root_out_dir}."
echo "Done."