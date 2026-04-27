"""
Summarises the completeness of stereo MOVi-E dataset splits by scanning each
dataset root for numeric sequence folders, reporting the total count, and
identifying any gaps (missing indices) in the expected contiguous range.
"""

import os

paths = {
    "lookat_linear_movement": "/mnt/Data/rajendra/kubric/generation/stereo_datasets/movi_e/lookat_orbit/linear_movement",
    "pure_translation": "/mnt/Data/rajendra/kubric/generation/stereo_datasets/movi_e/pure_translation",
}

for name, path in paths.items():
    folders = [int(f) for f in os.listdir(path) if f.isdigit()]
    folders.sort()

    if not folders:
        print(f"\n{name}: No numeric folders found")
        continue

    expected = set(range(min(folders), max(folders) + 1))
    actual = set(folders)
    missing = sorted(expected - actual)

    print(f"\nDataset: {name}")
    print(f"Total available = {len(folders)}")
    print(f"Missing folders: {missing}")
    print(f"Total missing = {len(missing)}")
