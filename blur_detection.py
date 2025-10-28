#!/usr/bin/env python3
"""
blur_detection.py

Scans a folder of images, computes four blur/sharpness metrics for each image,
computes z-scores for each metric, forms an equal-weight combined score, and
outputs a CSV table with: filename, raw metrics, z-values, combined score.

Usage:
    python blur_detection.py --folder /images --out results.csv --resize_max 1200 --center_crop 0.5

Requirements:
    pip install numpy opencv-python pandas tqdm pillow
"""
import os
import argparse
from glob import glob

import numpy as np
import cv2
import pandas as pd
from PIL import Image, ExifTags  # only used if you want to extend later
from tqdm import tqdm

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')

# -------------------------
# Metric implementations
# -------------------------
def variance_of_laplacian(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))

def tenengrad(gray, ksize=3):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    fm = gx**2 + gy**2
    return float(np.mean(fm))

def brenner(gray):
    h, w = gray.shape[:2]
    if w < 3:
        return 0.0
    diff = gray[:, 2:].astype(np.float64) - gray[:, :-2].astype(np.float64)
    return float((diff**2).sum() / max(diff.size, 1))

def fft_high_freq_energy(gray, keep_low_pct=0.05):
    # ratio of energy outside a central low-frequency rectangle
    f = np.fft.fft2(gray.astype(float))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    ly, lx = int(round(h * keep_low_pct / 2)), int(round(w * keep_low_pct / 2))
    ly = max(1, ly)
    lx = max(1, lx)
    low = mag[cy-ly:cy+ly+1, cx-lx:cx+lx+1].sum()
    total = mag.sum()
    if total == 0:
        return 0.0
    return float((total - low) / total)

# -------------------------
# Helpers
# -------------------------
def find_images(folder):
    paths = sorted(glob(os.path.join(folder, "*.*")))
    return [p for p in paths if p.lower().endswith(IMAGE_EXTS)]

def load_and_prep(path, resize_max=None, center_frac=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Can't read image: {path}")
    # resize (preserve aspect) if requested
    if resize_max and resize_max > 0:
        h, w = img.shape[:2]
        scale = float(resize_max) / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(round(w*scale)), int(round(h*scale))), interpolation=cv2.INTER_AREA)
    # center crop if requested
    if center_frac and (0 < center_frac < 1.0):
        h, w = img.shape[:2]
        new_h = int(round(h * center_frac))
        new_w = int(round(w * center_frac))
        y0 = (h - new_h) // 2
        x0 = (w - new_w) // 2
        img = img[y0:y0+new_h, x0:x0+new_w]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def zscores_list(values):
    arr = np.array(values, dtype=float)
    mu = arr.mean()
    sigma = arr.std(ddof=0)
    if sigma == 0:
        return [0.0] * len(arr)
    return ((arr - mu) / sigma).tolist()

# -------------------------
# Main processing
# -------------------------
def process_folder(folder, resize_max=None, center_frac=None, keep_low_pct=0.05, show_progress=True):
    image_paths = find_images(folder)
    if not image_paths:
        raise RuntimeError(f"No images found in {folder}")

    records = []
    iterator = tqdm(image_paths, desc="Processing images") if show_progress else image_paths
    for p in iterator:
        try:
            gray = load_and_prep(p, resize_max=resize_max, center_frac=center_frac)
            lap = variance_of_laplacian(gray)
            ten = tenengrad(gray)
            brn = brenner(gray)
            fft = fft_high_freq_energy(gray, keep_low_pct=keep_low_pct)
            records.append({
                "filename": os.path.basename(p),
                "path": p,
                "lap_var": lap,
                "tenengrad": ten,
                "brenner": brn,
                "fft_hf": fft
            })
        except Exception as e:
            print(f"ERROR processing {p}: {e}")

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("No successful metric computations.")

    # compute z-scores per metric
    df['z_lap'] = zscores_list(df['lap_var'].values)
    df['z_ten'] = zscores_list(df['tenengrad'].values)
    df['z_brenner'] = zscores_list(df['brenner'].values)
    df['z_fft'] = zscores_list(df['fft_hf'].values)

    # combined score = equal-weighted sum of z-scores (higher = sharper)
    df['score'] = 0.25 * (df['z_lap'] + df['z_ten'] + df['z_brenner'] + df['z_fft'])

    # sort descending by score
    df_sorted = df.sort_values(by='score', ascending=False).reset_index(drop=True)
    return df_sorted

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute z-scores and combined sharpness score for images in a folder.")
    parser.add_argument("--folder", required=True, help="Folder containing images.")
    parser.add_argument("--out", default="sharpness_table.csv", help="Output CSV path.")
    parser.add_argument("--resize_max", type=int, default=1200, help="Resize longest side to this px (0 to disable).")
    parser.add_argument("--center_crop", type=float, default=None, help="If set (0-1), compute metrics on center crop fraction.")
    parser.add_argument("--keep_low_pct", type=float, default=0.05, help="FFT low-frequency fraction to keep (default 0.05).")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar.")
    args = parser.parse_args()

    resize = args.resize_max if args.resize_max and args.resize_max > 0 else None
    show_progress = not args.no_progress

    df = process_folder(args.folder, resize_max=resize, center_frac=args.center_crop, keep_low_pct=args.keep_low_pct, show_progress=show_progress)

    # Keep only useful columns for the CSV (you can include more if desired)
    out_cols = ['filename', 'path', 'lap_var', 'tenengrad', 'brenner', 'fft_hf',
                'z_lap', 'z_ten', 'z_brenner', 'z_fft', 'score']
    df.to_csv(args.out, index=False, columns=out_cols)
    print(f"Saved results to {args.out}")
    print(df[out_cols].to_string(index=False))

if __name__ == "__main__":
    main()
