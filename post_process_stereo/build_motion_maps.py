import os
import os.path as osp
import argparse
from glob import glob
from tqdm import tqdm
from pathlib import Path
import json

import numpy as np
from camera_utils import get_cam_params, compute_motion_map
from io_utils import load_depth, load_optical_flow, save_motion_map, stitch_stereo_images_to_video

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_dir', type=str, required=True,
                        help='Path to the sequence directory containing frames and metadata.')
    parser.add_argument('--threshold', type=float, default=1,
                        help='Threshold for motion map computation.')
    parser.add_argument('--stitch_video', action='store_true',
                        help='If set, also stitch the motion maps into a video after processing.')
    args = parser.parse_args()

    # set paths (we will be using left camera data)
    for camera in ['left_camera', 'right_camera']:
        print(f'Processing camera: {camera}\n')
        depth_dir = osp.join(args.seq_dir, camera, 'depth')
        fflow_dir = osp.join(args.seq_dir, camera, 'forward_flow')
        bflow_dir = osp.join(args.seq_dir, camera, 'backward_flow')
        
        motion_map_dir = osp.join(args.seq_dir, camera, 'motion_map')
        os.makedirs(motion_map_dir, exist_ok=True)

        data_ranges_path = osp.join(args.seq_dir, camera, 'data_ranges.json')
        left_metadata_path = osp.join(args.seq_dir, camera, f'metadata_{camera}.json')

        # load necessary metadata
        left_metadata = json.load(open(left_metadata_path, 'r'))
        data_ranges = json.load(open(data_ranges_path, 'r'))
        fflow_max, fflow_min = data_ranges['forward_flow'].values()
        bflow_max, bflow_min = data_ranges['backward_flow'].values()

        # load the sequence file paths
        depth_paths = sorted(glob(osp.join(depth_dir, 'depth_*.tiff')))
        fflow_paths = sorted(glob(osp.join(fflow_dir, 'forward_flow_*.png')))
        bflow_paths = sorted(glob(osp.join(bflow_dir, 'backward_flow_*.png')))

        # load the camera parameters for the left camera
        K, E = get_cam_params(left_metadata).values()

        for i in tqdm(range(len(depth_paths)-1), desc='Generating motion maps', unit='frame', total=len(depth_paths)-1):

            # set cam params
            E0, E1 = E[i], E[i+1] # world → cam transformations
            cam_params0 = {'K': K, 'E': E0}
            cam_params1 = {'K': K, 'E': E1}

            # load depth maps
            depth0 = load_depth(depth_paths[i])
            fflow0 = load_optical_flow(fflow_paths[i], (fflow_min, fflow_max))
            bflow0 = load_optical_flow(bflow_paths[i+1], (bflow_min, bflow_max))

            motion_map, _ = compute_motion_map(cam_params0, cam_params1, depth0, fflow0, args.threshold)
            # motion_map = forward_backward_mask(fflow0, bflow0, args.threshold)
            motion_map_path = osp.join(motion_map_dir, f"{i:05d}_motion_map.png")
            save_motion_map(motion_map_path, motion_map)

    print('Motion maps for both cameras saved successfully.')

    if args.stitch_video:
        print('Stitching motion maps to video...')
        output_video_path = osp.join(args.seq_dir, 'motion_map_video.mp4')
        
        left_motion_map_dir = sorted(list((Path(args.seq_dir)/'left_camera'/'motion_map').glob('*.png')))
        right_motion_map_dir = sorted(list((Path(args.seq_dir)/'right_camera'/'motion_map').glob('*.png')))

        stitch_stereo_images_to_video(left_motion_map_dir, right_motion_map_dir, output_video_path, fps=12)
        print(f'Motion map video saved to: {output_video_path}')



