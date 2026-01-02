import os.path as osp
import glob
import json
import argparse
import cv2

def stitch_images_to_video(image_paths, output_path, fps): 
    """ Utility function to stitch a list of images into a video file.

    Args:
        image_paths: List of paths to image files.
        output_path: Path to save the output video file.
        fps: Frames per second for the output video.
    """
    import cv2

    if not image_paths:
        raise ValueError("No image paths provided for stitching.")

    # Read the first image to get dimensions
    first_image = cv2.imread(str(image_paths[0]))
    height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        video.write(img)

    video.release()


def stitch_stereo_images_to_video(left_image_paths, right_image_paths, output_path, fps):
    """Stitch left and right images side by side into a single video."""
    if not left_image_paths or not right_image_paths:
        raise ValueError("No image paths provided for stitching.")
    if len(left_image_paths) != len(right_image_paths):
        raise ValueError("Number of left and right images must be the same.")

    # Read the first images to get dimensions
    left_img = cv2.imread(str(left_image_paths[0]))
    right_img = cv2.imread(str(right_image_paths[0]))
    if left_img.shape != right_img.shape:
        raise ValueError("Left and right images must have the same shape.")
    height, width, layers = left_img.shape
    stereo_width = width * 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (stereo_width, height))

    for l_path, r_path in zip(left_image_paths, right_image_paths):
        l_img = cv2.imread(str(l_path))
        r_img = cv2.imread(str(r_path))
        stereo_img = cv2.hconcat([l_img, r_img])
        video.write(stereo_img)

    video.release()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate sample stereo videos from rendered images.')
    parser.add_argument('--sequence_dir', type=str, required=True,
                      help='Path to the directory containing sequence related data.')
    args = parser.parse_args()


    left_image_paths = sorted(glob.glob(osp.join(args.sequence_dir, 'left_camera', 'rgba', '*.png')))
    right_image_paths = sorted(glob.glob(osp.join(args.sequence_dir, 'right_camera', 'rgba', '*.png')))

    # extract frame rate
    metadata_path = osp.join(args.sequence_dir, "left_camera", "metadata_left_camera.json")
    metadata = json.load(open(metadata_path, 'r'))
    fps = metadata['flags']['frame_rate']
    sequence_name = osp.basename(args.sequence_dir).split('.')[0]

    # Stereo video (left | right)
    if left_image_paths and right_image_paths:
        stitch_stereo_images_to_video(
            left_image_paths=left_image_paths,
            right_image_paths=right_image_paths,
            output_path=osp.join(args.sequence_dir, f'{sequence_name}_stereo.mp4'),
            fps=fps
        )
        print(f"Stereo video (left | right) saved at: {osp.join(args.sequence_dir, f'{sequence_name}_stereo.mp4')}")
