#first run
# import os
# import sys
# import zipfile
# from pathlib import Path

# import numpy as np
# import requests
# from PIL import Image

# # CONFIG
# ZIP_FILE = "Dataset.zip"  # Path to the downloaded dataset zip
# DATASET_DIR = Path("Dataset")  # Unzipped folder
# TEMP_OUT_DIR = Path("submission_temp")  # Temporary folder for forged images
# FILE_PATH = "submission.zip"  # Final file to upload

# # Leaderboard submission
# BASE_URL  = "http://35.192.205.84:80"
# API_KEY  = "YOUR_API_KEY_HERE"  # REPLACE WITH YOUR API KEY
# TASK_ID   = "22-forging-task"

# # 1. UNZIP DATASET
# if not DATASET_DIR.exists():
#     if not os.path.exists(ZIP_FILE):
#         raise FileNotFoundError(f"Could not find {ZIP_FILE}. Please download the dataset first.")

#     print(f"Unzipping {ZIP_FILE}...")
#     with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
#         zip_ref.extractall(".")
# else:
#     print("Dataset already extracted.")

# # Ensure output directory exists
# TEMP_OUT_DIR.mkdir(exist_ok=True)


# # 2. NAIVE FORGERY ATTACK (IMAGE AVERAGING)
# print("Building forgery submission...")

# # Map the Dataset structure: (Source_Folder, Size_Subfolder, Target_Folder)
# CATEGORIES = [
#     ("WM_1", 1, 25),
#     ("WM_2", 26, 50),
#     ("WM_3", 51, 75),
#     ("WM_4", 76, 100),
#     ("WM_5", 101, 125),
#     ("WM_6", 126, 150),
#     ("WM_7", 151, 175),
#     ("WM_8", 176, 200),
# ]


# total_processed = 0

# for source_wm, target_start, target_stop in CATEGORIES:
#     print(f"Processing {source_wm} dataset -> Forging onto images {target_start}.png to {target_stop}.png ...")

#     source_dir = DATASET_DIR / "watermarked_sources" / source_wm
#     source_images = list(source_dir.glob("*.png"))

#     if not source_images:
#         print(f"  [Warning] No source images found in {source_dir}")
#         continue

#     target_dir = DATASET_DIR / "clean_targets"
#     target_images = []

#     for number in range(target_start, target_stop + 1, 1):
#         temp = target_dir / f"{number}.png"
#         target_images.append(temp)

#     for target_path, source_path in zip(target_images, source_images):
#         # Load target clean image
#         target_pil = Image.open(target_path).convert("RGB")

#         # Load target source image
#         source_pil = Image.open(source_path).convert("RGB")

#         # Convert to numpy arrays for the math
#         target_arr = np.array(target_pil).astype(np.float32)
#         source_arr = np.array(source_pil).astype(np.float32)

#         # Blend the Image with a Watermarked Image (Alpha Blending)
#         forged_img = (target_arr * 0.5) + (source_arr * 0.5)

#         # Clip values to valid pixel range [0, 255] and convert to uint8
#         forged_img = np.clip(forged_img, 0, 255).astype(np.uint8)

#         # Save to our temporary flat directory using the exact original filename (e.g., "104.png")
#         out_path = TEMP_OUT_DIR / target_path.name
#         Image.fromarray(forged_img).save(out_path)
#         total_processed += 1

# print(f"\nSuccessfully forged {total_processed} images.")
# if total_processed != 200:
#     print(f"[WARNING] Expected 200 images, but processed {total_processed}. Your submission may be rejected!")


# # 3. PACKAGE INTO FLAT ZIP FILE
# print(f"Packaging images into {FILE_PATH}...")
# with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
#     for img_path in TEMP_OUT_DIR.glob("*.png"):
#         zipf.write(img_path, arcname=img_path.name)

# print(f"Saved submission file to {FILE_PATH}")


#best one yet (0.333)
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter

# CONFIG
ZIP_FILE = "Dataset.zip"
DATASET_DIR = Path("Dataset")
TEMP_OUT_DIR = Path("submission_temp")
FILE_PATH = "submission.zip"

# Leaderboard submission
BASE_URL  = "http://35.192.205.84:80"
API_KEY  = "YOUR_API_KEY_HERE"
TASK_ID   = "22-forging-task"

# Global strength multiplier (THE knob to tune on the leaderboard). Each group's
# base gain is auto-computed so a forged target matches a genuine watermarked
# source's projection onto the extracted watermark (GLOBAL_SCALE=1.0). Values >1
# add detection margin at some LPIPS cost. Mind the 60-min cooldown.
GLOBAL_SCALE = 2.0

# Per-group extraction recipe, chosen by maximizing matched-filter separation
# (d-prime) of watermarked sources vs clean targets. Schemes leave their
# consistent signal in different frequency bands -> different recipes.
#   ('medhp', s)    img - median_filter(img, s)
#   ('ghp', s)      img - gaussian_blur(img, sigma=s)
#   ('dog', lo, hi) gaussian(lo) - gaussian(hi)
# Aggregation across the 25 sources = per-pixel MEDIAN (outlier-robust).
EXTRACT = {
    "WM_1": ("dog", 0.5, 2.0),
    "WM_2": ("medhp", 7),
    "WM_3": ("ghp", 1.0),
    "WM_4": ("medhp", 3),
    "WM_5": ("medhp", 3),
    "WM_6": ("medhp", 3),
    "WM_7": ("medhp", 5),
    "WM_8": ("medhp", 3),
}

# 1. UNZIP DATASET
if not DATASET_DIR.exists():
    if not os.path.exists(ZIP_FILE):
        raise FileNotFoundError(f"Could not find {ZIP_FILE}. Please download the dataset first.")
    print(f"Unzipping {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(".")
else:
    print("Dataset already extracted.")

TEMP_OUT_DIR.mkdir(exist_ok=True)

# 2. PER-GROUP WATERMARK EXTRACTION + PROJECTION-MATCHED COPY
print("Building forgery submission...")

CATEGORIES = [
    ("WM_1", 1, 25), ("WM_2", 26, 50), ("WM_3", 51, 75), ("WM_4", 76, 100),
    ("WM_5", 101, 125), ("WM_6", 126, 150), ("WM_7", 151, 175), ("WM_8", 176, 200),
]


def load(path):
    return np.array(Image.open(path).convert("RGB")).astype(np.float32)


def extract_watermark(imgs, spec):
    kind = spec[0]
    if kind == "medhp":
        s = spec[1]
        res = [im - median_filter(im, size=(s, s, 1)) for im in imgs]
    elif kind == "ghp":
        s = spec[1]
        res = [im - gaussian_filter(im, sigma=(s, s, 0)) for im in imgs]
    elif kind == "dog":
        lo, hi = spec[1], spec[2]
        res = [gaussian_filter(im, (lo, lo, 0)) - gaussian_filter(im, (hi, hi, 0)) for im in imgs]
    else:
        raise ValueError(spec)
    delta = np.median(res, axis=0)
    return delta - delta.mean()


def projection_gain(delta, sources, targets):
    v = delta.ravel()
    nv = np.linalg.norm(v) + 1e-9
    proj = lambda im: (im.ravel() - im.mean()) @ v / nv
    p_src = np.mean([proj(im) for im in sources])
    p_tgt = np.mean([proj(im) for im in targets])
    return (p_src - p_tgt) / nv


total_processed = 0

for source_wm, target_start, target_stop in CATEGORIES:
    print(f"Processing {source_wm} dataset -> Forging onto images {target_start}.png to {target_stop}.png ...")
    source_dir = DATASET_DIR / "watermarked_sources" / source_wm
    source_images = sorted(source_dir.glob("*.png"))
    if not source_images:
        print(f"  [Warning] No source images found in {source_dir}")
        continue

    target_dir = DATASET_DIR / "clean_targets"
    target_paths = [target_dir / f"{n}.png" for n in range(target_start, target_stop + 1)]

    srcs = [load(p) for p in source_images]
    tgts = [load(p) for p in target_paths]

    delta = extract_watermark(srcs, EXTRACT[source_wm])
    g = projection_gain(delta, srcs, tgts) * GLOBAL_SCALE

    for target_path, target_arr in zip(target_paths, tgts):
        wm = delta
        if target_arr.shape != wm.shape:
            resized = Image.fromarray(np.clip(wm + 128, 0, 255).astype(np.uint8)).resize(target_arr.shape[1::-1])
            wm = np.array(resized).astype(np.float32) - 128
        forged_img = np.clip(target_arr + g * wm, 0, 255).astype(np.uint8)
        Image.fromarray(forged_img).save(TEMP_OUT_DIR / target_path.name)
        total_processed += 1

print(f"\nSuccessfully forged {total_processed} images.")
if total_processed != 200:
    print(f"[WARNING] Expected 200 images, but processed {total_processed}. Your submission may be rejected!")

# 3. PACKAGE INTO FLAT ZIP FILE
print(f"Packaging images into {FILE_PATH}...")
with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for img_path in TEMP_OUT_DIR.glob("*.png"):
        zipf.write(img_path, arcname=img_path.name)
print(f"Saved submission file to {FILE_PATH}")