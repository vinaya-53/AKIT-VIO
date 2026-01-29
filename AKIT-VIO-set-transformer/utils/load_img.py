import os
import glob
import numpy as np
import torch
from PIL import Image
import pandas as pd

def load_tiff_images(folder="images/", max_images=50):
    image_paths = glob.glob(os.path.join(folder, "*.tif")) + glob.glob(os.path.join(folder, "*.tiff"))
    if len(image_paths) == 0:
        raise RuntimeError("No images found in " + folder)

    def extract_ts(path):
        name = os.path.basename(path)
        ts_raw = int(name.split("_")[0])
        return ts_raw

    image_paths = sorted(image_paths, key=extract_ts)[:max_images]

    # detect magnitude of raw timestamps and convert to seconds
    first_raw = extract_ts(image_paths[0])
    digits = len(str(first_raw))
    if digits >= 18:
        scale = 1e9   # nanoseconds -> seconds
    elif digits >= 15:
        scale = 1e6   # microseconds -> seconds
    elif digits >= 12:
        scale = 1e3   # milliseconds -> seconds
    else:
        scale = 1.0   # already seconds

    images = []
    timestamps = []
    for path in image_paths:
        fname = os.path.basename(path)
        ts_raw = int(fname.split("_")[0])
        ts_sec = ts_raw / scale
        timestamps.append(ts_sec)
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.tensor(arr).permute(2, 0, 1)  # [C,H,W]
        images.append(tensor)

    images = torch.stack(images)  # [N, 3, H, W]
    timestamps = torch.tensor(timestamps, dtype=torch.float64)
    return images, timestamps