"""Generate train/val split text files.

Scans the images/ directory, randomly assigns 90% of frames to train and
10% to val, then writes train.txt and val.txt (one filename per line,
without directory prefix or extension).

Usage:
    python generate_split.py [--data_root /path/to/XrayDepthEstScaling]
                             [--train_ratio 0.9] [--seed 42]
"""

import argparse
import glob
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
    )
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image_dir = os.path.join(args.data_root, "images")
    depth_dir = os.path.join(args.data_root, "depths")

    # Collect basenames that have both an image and a depth file.
    basenames = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(image_dir, "*.tif"))
        if os.path.exists(os.path.join(depth_dir, os.path.basename(p)))
    )

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(basenames))
    split = int(len(basenames) * args.train_ratio)

    train_names = sorted(basenames[i] for i in indices[:split])
    val_names = sorted(basenames[i] for i in indices[split:])

    train_path = os.path.join(args.data_root, "train.txt")
    val_path = os.path.join(args.data_root, "val.txt")

    with open(train_path, "w") as f:
        f.write("\n".join(train_names) + "\n")
    with open(val_path, "w") as f:
        f.write("\n".join(val_names) + "\n")

    print(f"Total: {len(basenames)}  Train: {len(train_names)}  Val: {len(val_names)}")
    print(f"Written: {train_path}")
    print(f"Written: {val_path}")


if __name__ == "__main__":
    main()
