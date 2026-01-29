import cv2
import png
import imageio
import numpy as np
from PIL import Image
from pathlib import Path

def load_image(path: str) -> np.ndarray:
    """Load an image from a file.

    Args:
        path (str): Path to the image file. 
    Returns:
        np.ndarray: Loaded image as a numpy array of shape (H, W, 3).
    """
    image = Image.open(path).convert('RGB')
    return np.array(image)[:, :, :3]  # (H, W, 3)

def load_depth(filename: str) -> np.ndarray:
    """Load depth map from a TIFF file.

    Args:
        path (str): Path to the depth map TIFF file.
    Returns:
        np.ndarray: Loaded depth map as a numpy array of shape (H, W,).
    """
    filename = Path(filename)
    img = imageio.imread(filename.read_bytes(), format="tiff")
    return img[:, :, :1] # (H, W)

def load_segmentation_map(filename: str) -> np.ndarray:
    """Load segmentation map from a PNG file.

    Args:
        path (str): Path to the segmentation map PNG file.
    Returns:
        np.ndarray: Loaded segmentation map as a numpy array of shape (H, W).
    """
    img = Image.open(filename)
    return np.array(img)  # (H, W)

def load_dynamic_mask(filename: str) -> np.ndarray:
    """Load dynamic mask from a PNG file.

    Args:
        path (str): Path to the dynamic mask PNG file.  
    Returns:
        np.ndarray: Loaded dynamic mask as a numpy array of shape (H, W), values in [0, 1].
    """
    img = Image.open(filename).convert('L')
    mask = np.array(img).astype(np.float32) / 255.0
    return mask  # (H, W)

def load_optical_flow(filename, rescale_range=None) -> np.ndarray:
  """
  Load optical flow from a PNG file.
  Note: The flow is stored in (delta_y, delta_x) order in the PNG.
  """
  png_reader = png.Reader(bytes=Path(filename).read_bytes())
  width, height, pngdata, info = png_reader.read()
  del png_reader

  bitdepth = info["bitdepth"]
  if bitdepth == 8:
    dtype = np.uint8
  elif bitdepth == 16:
    dtype = np.uint16
  else:
    raise NotImplementedError(f"Unsupported bitdepth: {bitdepth}")

  plane_count = info["planes"]
  pngdata = np.vstack(list(map(dtype, pngdata)))
  if rescale_range is not None:
    minv, maxv = rescale_range
    pngdata = pngdata / 2**bitdepth * (maxv - minv) + minv

  return pngdata.reshape((height, width, plane_count))[..., :2] # returns (delta_y, delta_x)

def save_motion_map(path: str, motion_map: np.ndarray):
    """Save motion map to a PNG file.

    Args:
        path (str): Path to save the motion map PNG file.
        motion_map (np.ndarray): Motion map as a numpy array of shape (H, W).
    """
    motion_map_uint8 = (motion_map * 255).astype(np.uint8)
    cv2.imwrite(path, motion_map_uint8)

def stitch_to_video(image_dir: str, output_path: str, fps: int = 30):
    """Stitch images in a directory to a video.

    Args:
        image_dir (str): Directory containing image files.
        output_path (str): Path to save the output video file.
        fps (int, optional): Frames per second for the output video. Defaults to 30.
    """
    image_paths = sorted(Path(image_dir).glob('*.png'))
    if not image_paths:
        raise ValueError(f"No images found in directory: {image_dir}")

    first_image = cv2.imread(str(image_paths[0]))
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        video_writer.write(frame)

    video_writer.release()

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

def stitch_stereo_depth_to_video(
    left_depth_paths,
    right_depth_paths,
    output_path,
    fps,
    percentile=(1, 99),
):
    """Stitch left and right depth TIFFs side by side into a single MP4 video."""

    if not left_depth_paths or not right_depth_paths:
        raise ValueError("No depth paths provided for stitching.")
    if len(left_depth_paths) != len(right_depth_paths):
        raise ValueError("Number of left and right depth maps must be the same.")

    # def load_depth(path):
    #     path = Path(path)
    #     img = imageio.imread(path.read_bytes(), format="tiff")
    #     return img[:, :, :1]  # (H, W, 1)

    # ---- compute global normalization bounds ----
    all_depths = []
    for l, r in zip(left_depth_paths, right_depth_paths):
        all_depths.append(load_depth(l).squeeze(-1).reshape(-1))
        all_depths.append(load_depth(r).squeeze(-1).reshape(-1))
    all_depths = np.concatenate(all_depths)
    d_min, d_max = np.percentile(all_depths, percentile)

    # ---- initialize video writer ----
    d0 = load_depth(left_depth_paths[0])
    H, W, _ = d0.shape
    stereo_width = W * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (stereo_width, H), isColor=True)

    # ---- write frames ----
    for l_path, r_path in zip(left_depth_paths, right_depth_paths):
        l_depth = load_depth(l_path).squeeze(-1)
        r_depth = load_depth(r_path).squeeze(-1)

        def normalize(depth):
            depth = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)
            return (depth * 255).astype(np.uint8)

        l_u8 = normalize(l_depth)
        r_u8 = normalize(r_depth)

        l_bgr = cv2.cvtColor(l_u8, cv2.COLOR_GRAY2BGR)
        r_bgr = cv2.cvtColor(r_u8, cv2.COLOR_GRAY2BGR)

        stereo_img = cv2.hconcat([l_bgr, r_bgr])
        video.write(stereo_img)

    video.release()
