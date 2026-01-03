import cv2
import numpy as np
from pathlib import Path
import argparse

def load_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def sift_rectification_validation(img_left, img_right,
                                  min_matches=50,
                                  ransac_thresh=1.0):
    """
    Validates stereo rectification using epipolar geometry.
    Measures point-to-epipolar-line vertical error in pixels.
    """

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img_left, None)
    kp2, des2 = sift.detectAndCompute(img_right, None)

    if des1 is None or des2 is None:
        print("No descriptors found.")
        return

    # FLANN matcher
    index_params = dict(algorithm=1, trees=5)  # KD-tree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    pts1 = []
    pts2 = []

    # Lowe ratio test
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    if len(pts1) < min_matches:
        print(f"Not enough good matches: {len(pts1)}")
        return

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    # Estimate Fundamental Matrix with RANSAC
    # F -> basically transforms points in image1 to epipolar lines in image2
    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=ransac_thresh,
        confidence=0.999
    )

    if F is None:
        print("Fundamental matrix estimation failed.")
        return

    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]

    if len(inliers1) < min_matches:
        print(f"Too few inliers after RANSAC: {len(inliers1)}")
        return

    # Compute epipolar lines in right image for left points
    lines2 = cv2.computeCorrespondEpilines(inliers1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)

    # Point-to-line distance
    # Calculate the vertical distance from each point in image2 to its corresponding epipolar line (calculated using points from image1)
    epipolar_errors = []
    for (x2, y2), (a, b, c) in zip(inliers2, lines2):
        dist = abs(a * x2 + b * y2 + c) / np.sqrt(a * a + b * b)
        epipolar_errors.append(dist)

    # Should be small for well-rectified images
    epipolar_errors = np.array(epipolar_errors)

    mean_err = np.mean(epipolar_errors)
    std_err  = np.std(epipolar_errors)
    p95      = np.percentile(epipolar_errors, 95)

    print(f"Inlier matches: {len(epipolar_errors)}")
    print(f"Mean epipolar error: {mean_err:.4f} px")
    print(f"Std dev: {std_err:.4f} px")
    print(f"95th percentile: {p95:.4f} px")

    # Proper rectification quality thresholds
    if mean_err < 0.3 and p95 < 1.0:
        print("Status: EXCELLENT RECTIFICATION")
    elif mean_err < 0.6 and p95 < 2.0:
        print("Status: GOOD RECTIFICATION")
    else:
        print("Status: POOR RECTIFICATION")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SIFT-based Epipolar Geometry Validation for Stereo Rectification")
    parser.add_argument("--seq_name", type=str, help="Sequence name/number")
    parser.add_argument("--stereo_type", type=str, choices=["pure_translation", "lookat_orbit"], help="Type of stereo setup")
    parser.add_argument("--camera_movement", type=str, default="", choices=["linear_movement", "linear_movement_linear_lookat"], help="Camera movement type (only for lookat_orbit)")
    parser.add_argument("-l", "--left_image", type=str, help="Path to left image")
    parser.add_argument("-r", "--right_image", type=str, help="Path to right image")
    args = parser.parse_args()

    if args.left_image and args.right_image:
        left_img = load_image(args.left_image)
        right_img = load_image(args.right_image)
        sift_rectification_validation(left_img, right_img)
        exit(0)

    if args.stereo_type != "lookat_orbit":
        sequence_dir = Path(f"/Users/ajay/Documents/Visual-Intelligence-Lab/MathWorks_Project/kubric/generation/stereo_datasets/movi_e/{args.stereo_type}/{args.seq_name}")
    else:
        sequence_dir = Path(f"/Users/ajay/Documents/Visual-Intelligence-Lab/MathWorks_Project/kubric/generation/stereo_datasets/movi_e/{args.stereo_type}/{args.camera_movement}/{args.seq_name}")

    left_images = sorted((sequence_dir / "left_camera" / "rgba").glob("rgba_*.png"))
    right_images = sorted((sequence_dir / "right_camera" / "rgba").glob("rgba_*.png"))
    print(f"Total Frames to Process: {len(left_images)}")

    for left_img_path, right_img_path in zip(left_images, right_images):
        left_img = load_image(str(left_img_path))
        right_img = load_image(str(right_img_path))
        print(f"Processing Frame: {left_img_path.name}")
        sift_rectification_validation(left_img, right_img)
        print("-" * 50)