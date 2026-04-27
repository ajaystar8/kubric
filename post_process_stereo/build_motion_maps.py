"""
Generates per-frame dynamic object masks and dynamic flow magnitude maps for
both cameras in a stereo sequence, using depth maps, optical flow, and camera
extrinsics to separate scene motion caused by camera movement from motion caused
by independently moving objects.

For each consecutive frame pair, the script:
  1. Loads depth, forward flow (frame i), and backward flow (frame i+1).
  2. Uses the camera intrinsics (K) and extrinsics (E) to compute the expected
     camera-induced optical flow via ``compute_motion_map``, then compares it
     against the observed flow to isolate dynamic (non-static) regions.
  3. Optionally refines the binary dynamic mask using instance segmentation maps
     (``--refine``) to snap mask boundaries to object contours.
  4. Saves each binary mask as ``dynamic_mask_NNNNN.png`` (0/255 uint8) and a
     plasma-colormap visualization of the dynamic flow magnitude as
     ``dynamic_flow_NNNNN.png`` under ``<seq_dir>/<camera>/dynamic_masks/`` and
     ``<seq_dir>/<camera>/dynamic_flows/``, respectively.

Optionally (``--stitch_video``), stitches left/right results into side-by-side
stereo MP4 videos at 12 fps for: dynamic masks, dynamic flows, RGBA frames,
forward flow, and depth maps.

Usage:
    python build_motion_maps.py --seq_dir <path> [--threshold <float>]
                                [--refine] [--stitch_video]

Arguments:
    --seq_dir       Root directory of the sequence; must contain
                    ``left_camera/`` and ``right_camera/`` subdirectories with
                    depth, forward_flow, backward_flow, segmentation, and
                    metadata files.
    --threshold     Motion magnitude threshold for classifying a pixel as
                    dynamic (default: 1).
    --refine        If set, refines motion masks using instance segmentation.
    --stitch_video  If set, produces stereo side-by-side MP4 videos after
                    processing.
"""

import os
import os.path as osp
import argparse
from glob import glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import json

import numpy as np
from camera_utils import get_cam_params, compute_motion_map, refine_motion_map
from io_utils import load_depth, load_optical_flow, load_segmentation_map, stitch_stereo_images_to_video, stitch_stereo_depth_to_video

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_dir', type=str, required=True,
                        help='Path to the sequence directory containing frames and metadata.')
    parser.add_argument('--threshold', type=float, default=1,
                        help='Threshold for motion map computation.')
    parser.add_argument('--refine', action='store_true',
                        help='If set, refine motion maps using segmentation masks.')
    parser.add_argument('--stitch_video', action='store_true',
                        help='If set, also stitch the motion maps into a video after processing.')
    args = parser.parse_args()

    # set paths (we will be using left camera data)
    for camera in ['left_camera', 'right_camera']:
        print(f'Processing camera: {camera}\n')
        depth_dir = osp.join(args.seq_dir, camera, 'depth')
        fflow_dir = osp.join(args.seq_dir, camera, 'forward_flow')
        bflow_dir = osp.join(args.seq_dir, camera, 'backward_flow')
        segmentation_dir = osp.join(args.seq_dir, camera, 'segmentation')
        
        dynamic_masks_dir = osp.join(args.seq_dir, camera, "dynamic_masks")
        dynamic_flows_dir = osp.join(args.seq_dir, camera, "dynamic_flows")
        os.makedirs(dynamic_masks_dir, exist_ok=True)
        os.makedirs(dynamic_flows_dir, exist_ok=True)

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
        segmentation_paths = sorted(glob(osp.join(segmentation_dir, '*.png')))

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

            dynamic_map, mag_dynamic_flow = compute_motion_map(cam_params0, cam_params1, depth0, fflow0, args.threshold)

            # save plasma colormap of magnitude of dynamic flow
            plt.imsave(osp.join(dynamic_flows_dir, f"dynamic_flow_{i:05d}.png"), mag_dynamic_flow)

            if args.refine:
                instance_map = load_segmentation_map(segmentation_paths[i])
                dynamic_map = refine_motion_map(dynamic_map.astype(bool), instance_map)
            
            dynamic_mask_uint8 = (dynamic_map * 255).astype(np.uint8)
            out_path = osp.join(dynamic_masks_dir, f"dynamic_mask_{i:05d}.png")
            cv2.imwrite(out_path, dynamic_mask_uint8)

    print('Motion maps for both cameras saved successfully.')

    if args.stitch_video:
        print('Stitching stereo maps into videos...')

        # stitch stereo videos
        stitch_stereo_images_to_video(
            sorted(glob(osp.join(args.seq_dir, "left_camera", "dynamic_masks", "*.png"))),
            sorted(glob(osp.join(args.seq_dir, "right_camera", "dynamic_masks", "*.png"))),
            osp.join(args.seq_dir, "dynamic_masks_stereo.mp4"),
            fps=12
        )

        stitch_stereo_images_to_video(
            sorted(glob(osp.join(args.seq_dir, "left_camera", "dynamic_flows", "*.png"))),
            sorted(glob(osp.join(args.seq_dir, "right_camera", "dynamic_flows", "*.png"))),
            osp.join(args.seq_dir, "dynamic_flows_stereo.mp4"),
            fps=12
        )

        stitch_stereo_images_to_video(
            sorted(glob(osp.join(args.seq_dir, "left_camera", "rgba", "*.png"))),
            sorted(glob(osp.join(args.seq_dir, "right_camera", "rgba", "*.png"))),
            osp.join(args.seq_dir, "rgba_stereo.mp4"),
            fps=12
        )

        stitch_stereo_images_to_video(
            sorted(glob(osp.join(args.seq_dir, "left_camera", "forward_flow", "*.png"))),
            sorted(glob(osp.join(args.seq_dir, "right_camera", "forward_flow", "*.png"))),
            osp.join(args.seq_dir, "forward_flow_stereo.mp4"),
            fps=12
        )

        # stitch stereo videos
        stitch_stereo_depth_to_video(
            sorted(glob(osp.join(args.seq_dir, "left_camera", "depth", "*.tiff"))),
            sorted(glob(osp.join(args.seq_dir, "right_camera", "depth", "*.tiff"))),
            osp.join(args.seq_dir, "depth_maps_stereo.mp4"),
            fps=12
        )

        print("Done!")


