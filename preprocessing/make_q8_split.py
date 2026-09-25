#!/usr/bin/env python3
"""Build an 8-bit-input twin of a test split, so every method is fed the same
thing.

The twin replaces the split's float32 guidance with `srgb_to_linear(png / 255)`,
where `png` is the split's own `ldr_input` image, the one other methods read.

    python preprocessing/make_q8_split.py --src <split> --dst <split>_q8 \
        [--file_list SIHDR_test.txt]

The target folder and `ldr_input` are symlinked rather than copied, and the
index is copied verbatim.

Sample from this split, but align and score against the original.
"""
import argparse
import os
import shutil

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="the split to mirror")
ap.add_argument("--dst", required=True, help="the 8-bit twin to create")
ap.add_argument("--file_list", default="SIHDR_test.txt",
                help="index file inside --src")
a = ap.parse_args()

src, dst = os.path.abspath(a.src), os.path.abspath(a.dst)
if os.path.exists(dst) and os.listdir(dst):
    raise SystemExit(f"ABORT: {dst} exists and is not empty")

# The folder names come from the index's own contents, not from the index
# filename.
with open(os.path.join(src, a.file_list)) as f:
    first = next(line for line in f if line.strip())
t_rel, g_rel = first.strip().split(",")[:2]
tgt_dirname = os.path.dirname(t_rel)
gui_dirname = os.path.dirname(g_rel)

dg = os.path.join(dst, gui_dirname)
os.makedirs(dg, exist_ok=True)

# Relative symlinks to the source's target folder and ldr_input.
for link in (tgt_dirname, "ldr_input"):
    p = os.path.join(dst, link)
    if not os.path.exists(p):
        os.symlink(os.path.join(os.path.relpath(src, dst), link), p)
shutil.copy(os.path.join(src, a.file_list), os.path.join(dst, a.file_list))

names = []
with open(os.path.join(src, a.file_list)) as f:
    for line in f:
        line = line.strip()
        if line:
            names.append(os.path.splitext(os.path.basename(line.split(",")[1]))[0])

n, worst = 0, 0.0
for nm in names:
    png = os.path.join(src, "ldr_input", nm + ".png")
    p = np.asarray(Image.open(png).convert("RGB")).astype(np.float64) / 255.0
    lin = np.where(p <= 0.04045, p / 12.92, ((p + 0.055) / 1.055) ** 2.4)
    lin = np.clip(lin, 0.0, 1.0).astype(np.float32)
    np.save(os.path.join(dg, nm + ".npy"), lin)
    old = np.load(os.path.join(src, gui_dirname, nm + ".npy"))
    worst = max(worst, float(np.abs(old.astype(np.float64) - lin).max()))
    n += 1
print(f"wrote {n} guidance npy -> {dg}")
print(f"max |float - quantized| over the split: {worst:.6f}")
