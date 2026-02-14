import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop


class XrayDepth(Dataset):
    """PyTorch dataset for X-ray depth estimation from C-arm CBCT data.

    Expected directory layout under `data_root`:
        images/  - 768x768 float32 grayscale TIFF (values in [0, 1])
        depths/  - 768x768 uint16 TIFF (0 = invalid/background)
        poses/   - JSON camera pose metadata

    Filenames match across the three directories, e.g.:
        images/Pelvic-Ref-001_0001.tif
        depths/Pelvic-Ref-001_0001.tif
        poses/Pelvic-Ref-001_0001.json
    """

    def __init__(
        self,
        data_root,
        filelist_path,
        mode,
        size=(518, 518),
        max_depth=1000.0,
    ):
        """
        Args:
            data_root: Root directory containing images/ and depths/ folders.
            filelist_path: Path to a split file (e.g. train.txt or val.txt)
                           with one basename per line (no directory, no extension).
            mode: 'train' or 'eval'.
            size: Network input size (width, height).
            max_depth: Maximum valid depth in metres after conversion.
                       Depths beyond this are masked out.  Raw uint16 values
                       are stored at 0.1 mm resolution (1 unit = 0.1 mm), so
                       the typical range ~3000-8000 maps to ~300.0-800.0 millimetres.
        """
        self.mode = mode
        self.size = size
        self.max_depth = max_depth
        self.data_root = data_root

        with open(filelist_path, "r") as f:
            basenames = [line.strip() for line in f if line.strip()]

        image_dir = os.path.join(data_root, "images")
        depth_dir = os.path.join(data_root, "depths")
        self.filelist = [
            f"{os.path.join(image_dir, b + '.tif')} {os.path.join(depth_dir, b + '.tif')}"
            for b in basenames
        ]

        net_w, net_h = size
        self.transform = Compose(
            [
                Resize(
                    width=net_w,
                    height=net_h,
                    resize_target=True if mode == "train" else False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
            + ([Crop(size[0])] if self.mode == "train" else [])
        )

    def __getitem__(self, item):
        img_path = self.filelist[item].split(" ")[0]
        depth_path = self.filelist[item].split(" ")[1]

        # --- Image ---
        # X-ray images are single-channel float32 TIFF in [0, 1].
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # (H, W) float32
        # Replicate to 3 channels (DINOv2 backbone expects RGB).
        image = np.stack([image, image, image], axis=-1)  # (H, W, 3)
        # Already [0, 1] so no /255 needed.
        image = image.astype(np.float64)

        # --- Depth ---
        # Depth maps are uint16 TIFF; 1 unit = 0.1 mm, 0 = invalid/background.
        # Divide by 10 to convert to millimetres
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32) / 10.0

        sample = self.transform({"image": image, "depth": depth})

        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])

        sample["valid_mask"] = (sample["depth"] > 0) & (
            sample["depth"] <= self.max_depth
        )

        sample["image_path"] = img_path

        return sample

    def __len__(self):
        return len(self.filelist)
