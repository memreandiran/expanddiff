#!/usr/bin/env python3
"""FID-R: FID on Reinhard tone-mapped crops, following LEDiff (Wang et al., CVPR
2025), which follows Chai et al. (ECCV 2022).

60 random 128x128 crops per test image, pooled over the split, embedded with
Inception and compared by Frechet distance against 20,000 reference crops drawn
at independent locations from the same targets.

  python metrics/compute_fid.py --device cuda \
      --data_dir <split> --file_list SIHDR_test.txt \
      --pred_dir <split>/<arm>/pred \
      --crop_size 128 --crops_per_image 60 --seed 1234 \
      --reference_crops 20000 --reference_seed 99991 --out fid.yaml

Average over --seed 1234, 7 and 99; the crop sample moves FID by an amount
comparable to small differences between methods. Keep --reference_seed fixed so
the reference crop set is identical across those runs.

--device defaults to cuda; pass cpu explicitly off-GPU. If torch-fidelity cannot
be constructed the script falls back to torchvision's inception_v3 and records
`feature_extractor` in the output yaml. The two are not comparable, so check
that field before using a value.

Relative paths are resolved against $EXPANDIFF_ROOT; absolute paths bypass that.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

# Only used to resolve RELATIVE --pred_dir / --data_dir / --out arguments.
# Pass absolute paths and it is never consulted.
ROOT = os.environ.get("EXPANDIFF_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RD = ROOT

PU_DENOM = float(np.log10(319.0))

CH_ART_SHADOW_ANY = 4
CH_ART_HIGHLIGHT_ANY = 6


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def reinhard(x, key=0.18, delta=1e-6):
    """Reinhard et al. (2002) global photographic operator, then sRGB gamma.

    This exists to bridge to LEDiff's Table 1, whose FID-R column tone-maps
    with Reinhard before computing FID. Their other two columns use Durand
    (bilateral, local) and Liang (L1-L0 layer decomposition); neither is
    reimplemented here, and a substitute would not be comparable, so only the
    -R column is offered.

    Note the key normalisation is PER IMAGE, so prediction and ground truth are
    each auto-exposed independently. That cancels absolute-scale error, which
    is what makes the operator usable across scene- and display-referred data
    -- but it also means this encoding cannot see a pure brightness offset.
    LEDiff's numbers carry the same property, so the comparison stays fair.
    """
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, None)
    lw = x @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    log_avg = np.exp(np.mean(np.log(delta + lw)))
    if not np.isfinite(log_avg) or log_avg <= 0:
        log_avg = delta
    scale = key / log_avg
    ls = lw * scale
    l_white = float(ls.max())
    if not np.isfinite(l_white) or l_white <= 0:
        l_white = 1.0
    ld = ls * (1.0 + ls / (l_white ** 2)) / (1.0 + ls)
    ratio = np.where(lw > delta, ld / np.maximum(lw, delta), 0.0)[..., None]
    return linear_to_srgb(np.clip(x * ratio, 0.0, 1.0))


ENCODERS = {"reinhard": reinhard}


def hwc(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[2] != 3:
        a = np.transpose(a, (1, 2, 0))
    return a


def _box_coverage(mask, ys, xs, crop):
    """Fraction of True pixels inside each [y:y+crop, x:x+crop] box, via an
    integral image so all candidates cost one gather each."""
    ii = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)
    y0, x0, y1, x1 = ys, xs, ys + crop, xs + crop
    total = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    return total / float(crop * crop)


def sample_boxes(h, w, n, crop, rng, mask=None, min_coverage=0.02, oversample=40):
    """n (y, x) top-left corners, plus the per-box clip coverage.

    With a mask, boxes are biased to regions where clipping actually happened:
    candidates are drawn uniformly, then filtered to coverage >=
    min_coverage. If too few qualify we fall back to the highest-coverage
    candidates -- the caller counts those, because a run where most boxes fell
    back is measuring mostly-unclipped content and the number is not the
    hallucination signal it claims to be. Deterministic given `rng`."""
    if h < crop or w < crop:
        return [], np.zeros(0)
    if mask is None:
        ys = rng.integers(0, h - crop + 1, size=n)
        xs = rng.integers(0, w - crop + 1, size=n)
        return list(zip(ys, xs)), np.full(n, np.nan)
    cand = int(n * oversample)
    ys = rng.integers(0, h - crop + 1, size=cand)
    xs = rng.integers(0, w - crop + 1, size=cand)
    cov = _box_coverage(mask, ys, xs, crop)
    keep = np.flatnonzero(cov >= min_coverage)
    sel = keep[:n] if keep.size >= n else np.argsort(-cov)[:n]
    return [(int(ys[i]), int(xs[i])) for i in sel], cov[sel]


class InceptionFeatures:
    """2048-d pool features. Prefers torch-fidelity's TF-ported weights (the
    canonical FID network); falls back to torchvision's inception_v3, which is
    self-consistent but on a different weight set -- so the extractor name is
    recorded in the output and numbers from the two must never be mixed."""

    def __init__(self, device="cuda"):
        import torch

        self.torch = torch
        self.device = device
        try:
            from torch_fidelity.feature_extractor_inceptionv3 import (
                FeatureExtractorInceptionV3,
            )

            self.net = FeatureExtractorInceptionV3(
                "inception-v3-compat", features_list=["2048"]
            ).to(device).eval()
            self.kind = "torch-fidelity-inception-v3-compat"
            self._fwd = self._fwd_fidelity
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] torch-fidelity unavailable ({exc.__class__.__name__}); "
                  "falling back to torchvision inception_v3. Numbers are "
                  "self-consistent but not canonical FID.", file=sys.stderr)
            import torchvision

            net = torchvision.models.inception_v3(weights="IMAGENET1K_V1",
                                                  aux_logits=True)
            net.fc = torch.nn.Identity()
            self.net = net.to(device).eval()
            self.kind = "torchvision-inception-v3-IMAGENET1K_V1"
            self._fwd = self._fwd_torchvision
        for p in self.net.parameters():
            p.requires_grad_(False)

    def _fwd_fidelity(self, u8):
        return self.net(u8)[0].double()

    def _fwd_torchvision(self, u8):
        torch = self.torch
        x = u8.float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(299, 299), mode="bilinear",
                                            align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return self.net((x - mean) / std).double()

    def __call__(self, patches_u8, batch_size=50):
        """patches_u8: (N, H, W, 3) uint8 -> (N, 2048) float64 numpy."""
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(patches_u8), batch_size):
                chunk = patches_u8[i:i + batch_size]
                t = torch.from_numpy(np.ascontiguousarray(
                    chunk.transpose(0, 3, 1, 2))).to(self.device)
                out.append(self._fwd(t).cpu().numpy())
        if not out:
            return np.zeros((0, 2048), dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32)


def frechet_distance(f_real, f_fake, eps=1e-6):
    """Standard FID: ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^(1/2))."""
    from scipy import linalg

    f_real = np.asarray(f_real, dtype=np.float64)
    f_fake = np.asarray(f_fake, dtype=np.float64)
    mu_r, mu_g = f_real.mean(axis=0), f_fake.mean(axis=0)
    s_r = np.cov(f_real, rowvar=False)
    s_g = np.cov(f_fake, rowvar=False)
    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(s_r.dot(s_g), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(s_r.shape[0]) * eps
        covmean = linalg.sqrtm((s_r + offset).dot(s_g + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s_r) + np.trace(s_g)
                 - 2.0 * np.trace(covmean))


def read_pairs(data_dir, file_list):
    pairs = []
    with open(os.path.join(data_dir, file_list)) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t, g = line.split(",")
            pairs.append((t, g))
    return pairs


def default_mask_eps(data_dir):
    """eps for clipmasks: 0.0, i.e. clipped means EXACTLY black (0) or EXACTLY
    white (1.0 == 255/255).

    The stored `_clipmasks.npy` on disk were written with eps=2/255 (verified
    bit-for-bit on both FiveK and HDR+ splits), but that tolerance is an 8-bit
    quantization allowance and is wrong for float32 linear data: it treats a
    target pixel at 0.003 as "already black" and therefore drops it from the
    artificial-clip mask, even though the clip genuinely destroyed signal
    there. In linear space 0.003 is dark, not black.

    So this script recomputes masks at eps=0.0 by default rather than reading
    the stored ones -- which also keeps a single definition across every split
    in a sweep, instead of mixing stored (2/255) and recomputed masks. Pass
    --mask_source stored to read the on-disk masks instead.

    The choice matters: on HDR+ `-12_-4_-4_0`, eps=0 gives 0.033 shadow
    coverage where the stored 2/255 mask gives 0.000.
    """
    return 0.0


def load_clipmask(data_dir, guidance_rel, target_rel, region, eps,
                  mask_source="recompute"):
    """The art_{shadow,highlight}_any mask for `region`.

    Default is to recompute from target+guidance at `eps` (see
    default_mask_eps for why the stored masks are not used), with the same
    structure as sample.py:compute_clipmasks_fallback -- artificial clipping is
    "guidance at the limit AND target not at the limit", reduced over channels
    with .any(). mask_source='stored' reads the preprocessed 8-channel file
    instead, falling back to recompute where absent (some splits, e.g.
    hdrplus_linrgb_full_-12_-6_-2_0_d4, predate the mask writer)."""
    if region == "all":
        return None
    if mask_source == "stored":
        p = os.path.join(data_dir, guidance_rel[:-4] + "_clipmasks.npy")
        if os.path.exists(p):
            ch = CH_ART_SHADOW_ANY if region == "shadow" else CH_ART_HIGHLIGHT_ANY
            return np.load(p).astype(bool)[ch]
    g = hwc(np.load(os.path.join(data_dir, guidance_rel)))
    t = hwc(np.load(os.path.join(data_dir, target_rel)))
    if region == "shadow":
        art_pc = (g <= eps) & ~(t <= eps)
    else:
        art_pc = (g >= 1.0 - eps) & ~(t >= 1.0 - eps)
    return art_pc.any(axis=-1)


_REF_CACHE = {}


def reference_family(data_dir):
    """Dataset identity for reference sharing, with the clipping stops stripped
    out of the split name.

    The unclipped TARGETS are identical across every stop variant of a dataset
    (verified: same 32,681 FiveK / 30,052 HDR+ target lists, byte-identical
    pixels), because the stops only change the guidance. So all FiveK splits
    share one reference and all HDR+ splits share another, instead of
    rebuilding the same 20k crops 15 times. The downsample suffix stays in the
    key so a d2 and a d4 variant never share.
    """
    base = os.path.basename(data_dir.rstrip("/"))
    num = r"-?\d+(?:\.\d+)?"
    return re.sub(rf"_{num}_{num}_{num}_{num}", "", base)


def reference_indices(data_dir, source):
    """Index files feeding the domain reference. source='all' uses every split
    available (train + test; REED also has dev), source='train' only train."""
    fams = (("HDRPlus_train.txt", "HDRPlus_test.txt"),
            ("REED_train.txt", "REED_dev.txt", "REED_test.txt"),
            ("Fairchild_test.txt",),
            ("SIHDR_test.txt",))
    for fam in fams:
        found = [f for f in fam if os.path.exists(os.path.join(data_dir, f))]
        if found:
            if source == "train":
                tr = [f for f in found if "_train" in f]
                return tr or found
            return found
    return []


def domain_reference_features(data_dir, encoding, n_crops, crop_size,
                              extractor, seed, batch_size, train_list=None,
                              source="all"):
    """Unpaired reference for the PURELY GENERATIVE framing: Inception features
    of real target patches drawn from the split's TRAIN targets.

    This answers "do the outputs look like real photographs of this domain",
    with no scene correspondence checked at all. It is the standard
    generative-model protocol: reference statistics come from the dataset, not
    from the specific images the model was run on.

    Train targets are used (not test) so the reference is disjoint from the
    scenes being scored, and because there are far more of them: 30-33k tiled
    256x256 patches per split, which gives a much better conditioned 2048x2048
    covariance than the 2940-6000 crops the paired mode can muster.

    Cached per (split, encoding, n_crops, seed) so every arm of a split is
    scored against the identical reference.
    """
    fam = reference_family(data_dir)
    key = (fam, encoding, n_crops, crop_size, seed, source, train_list)
    if key in _REF_CACHE:
        print(f"  reusing domain reference for family {fam} (enc={encoding})",
              flush=True)
        return _REF_CACHE[key]
    lists = [train_list] if train_list else reference_indices(data_dir, source)
    if not lists:
        raise SystemExit(f"[error] no index files in {data_dir} to build the "
                         f"reference crop set from")
    pairs = []
    for L in lists:
        pairs.extend(read_pairs(data_dir, L))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    encode = ENCODERS[encoding]
    per = max(1, int(np.ceil(n_crops / len(pairs))))
    feats, taken = [], 0
    print(f"  building domain reference for family {fam}: {n_crops} crops of "
          f"{crop_size} from {len(pairs)} real targets across "
          f"{'+'.join(lists)} ({per}/patch), enc={encoding}", flush=True)
    for j in order:
        if taken >= n_crops:
            break
        p = os.path.join(data_dir, pairs[j][0])
        if not os.path.exists(p):
            continue
        arr = np.clip(encode(hwc(np.load(p))), 0, 1).astype(np.float32)
        h, w = arr.shape[:2]
        if h < crop_size or w < crop_size:
            continue
        k = min(per, n_crops - taken)
        ys = rng.integers(0, h - crop_size + 1, size=k)
        xs = rng.integers(0, w - crop_size + 1, size=k)
        batch = np.empty((k, crop_size, crop_size, 3), dtype=np.uint8)
        for i, (y, x) in enumerate(zip(ys, xs)):
            batch[i] = np.rint(arr[y:y + crop_size, x:x + crop_size] * 255)
        feats.append(extractor(batch, batch_size))
        taken += k
        if taken % 5000 < per:
            print(f"    reference {taken}/{n_crops}", flush=True)
    out = np.concatenate(feats, axis=0)
    print(f"  domain reference ready: {len(out)} crops", flush=True)
    _REF_CACHE[key] = out
    return out


def _run_dir(pred_dir):
    """Walk up from visualizations/... to the sampling-run dir that holds
    metric.yaml, so fid.yaml lands next to it."""
    d = pred_dir.rstrip("/")
    for _ in range(3):
        d = os.path.dirname(d)
        if os.path.exists(os.path.join(d, "metric.yaml")):
            return d
    return pred_dir


def run_region(args, pairs, region, extractor):
    """Single-region convenience wrapper (used by the CLI)."""
    return run_regions(args, pairs, [region], extractor)[region]


def run_regions(args, pairs, regions, extractor):
    """FID for several regions in ONE pass over the images.

    Each image's prediction TIFF and GT .npy are ~9 MB, so reloading them per
    region would triple the read volume over the network PVC (measured: 259 GB
    across a full 65-arm sweep vs 92 GB sharing the pass). Masks for all
    regions are derived from a single guidance load too.
    """
    encs = {r: (("pu" if r == "shadow" else "srgb") if args.encoding == "auto"
                else args.encoding) for r in regions}
    st = {r: {"real": [], "fake": [], "n_img": 0, "n_skip_mask": 0,
              "n_skip_missing": 0, "cov": []} for r in regions}
    mask_source = getattr(args, "mask_source", "recompute")
    reference = getattr(args, "reference", "paired")
    tdir = args.data_dir
    c = args.crop_size
    if reference == "domain":
        for r in regions:
            st[r]["real_fixed"] = domain_reference_features(
                tdir, encs[r], args.reference_crops, c, extractor,
                args.reference_seed, args.batch_size,
                getattr(args, "reference_list", None),
                getattr(args, "reference_source", "all"))

    for idx, (t_rel, g_rel) in enumerate(pairs):
        t_path = os.path.join(tdir, t_rel)
        name = os.path.splitext(os.path.basename(t_rel))[0]

        p_path = os.path.join(args.pred_dir, name + "_generated_linear.tiff")
        if not (os.path.exists(t_path) and os.path.exists(p_path)):
            for r in regions:
                st[r]["n_skip_missing"] += 1
            continue
        import tifffile

        pred = hwc(tifffile.imread(p_path))

        gt = hwc(np.load(t_path))
        if pred.shape[:2] != gt.shape[:2]:
            print(f"[warn] shape mismatch {name}: {pred.shape} vs {gt.shape}, "
                  f"skipping", file=sys.stderr)
            for r in regions:
                st[r]["n_skip_missing"] += 1
            continue

        h, w = gt.shape[:2]
        enc_cache = ({e: np.clip(ENCODERS[e](gt), 0, 1).astype(np.float32)
                      for e in set(encs.values())}
                     if reference == "paired" else {})
        enc_pred = {e: np.clip(ENCODERS[e](pred), 0, 1).astype(np.float32)
                    for e in set(encs.values())}
        del gt, pred

        for r in regions:
            mask = load_clipmask(tdir, g_rel, t_rel, r, args._mask_eps, mask_source)
            if mask is not None:
                if mask.shape != (h, w):
                    print(f"[warn] mask shape {mask.shape} != image {(h, w)} "
                          f"for {name}, skipping", file=sys.stderr)
                    st[r]["n_skip_missing"] += 1
                    continue
                if not mask.any():
                    st[r]["n_skip_mask"] += 1
                    continue

            rng = np.random.default_rng(args.seed + idx)
            boxes, cov = sample_boxes(h, w, args.crops_per_image, c, rng,
                                      mask=mask, min_coverage=args.min_coverage)
            if not boxes:
                st[r]["n_skip_mask"] += 1
                continue
            st[r]["cov"].append(cov)

            g_enc = enc_cache.get(encs[r])
            p_enc = enc_pred[encs[r]]
            fake_p = np.empty((len(boxes), c, c, 3), dtype=np.uint8)
            for k, (y, x) in enumerate(boxes):
                fake_p[k] = np.rint(p_enc[y:y + c, x:x + c] * 255)
            st[r]["fake"].append(extractor(fake_p, args.batch_size))

            if reference == "paired":
                real_p = np.empty((len(boxes), c, c, 3), dtype=np.uint8)
                for k, (y, x) in enumerate(boxes):
                    real_p[k] = np.rint(g_enc[y:y + c, x:x + c] * 255)
                st[r]["real"].append(extractor(real_p, args.batch_size))
            st[r]["n_img"] += 1

        del enc_cache, enc_pred
        done = st[regions[0]]["n_img"]
        if done and done % 10 == 0:
            counts = " ".join(f"{r}={st[r]['n_img']}" for r in regions)
            print(f"  [{idx + 1}/{len(pairs)}] {counts}", flush=True)

    out = {}
    for r in regions:
        try:
            out[r] = _finalize(args, r, encs[r], st[r], extractor, reference)
        except ValueError as e:
            print(f"[skip] region={r}: {e}", file=sys.stderr)
            out[r] = {"fid": None, "skipped": str(e), "region": r,
                      "encoding": encs[r], "n_images": int(st[r]["n_img"])}
    return out


def _finalize(args, region, encoding, s, extractor, reference="paired"):
    if s["n_img"] < 2:
        raise ValueError(f"only {s['n_img']} usable images for region={region}")
    f_real = (s["real_fixed"] if reference == "domain"
              else np.concatenate(s["real"], axis=0))
    f_fake = np.concatenate(s["fake"], axis=0)
    fid = frechet_distance(f_real, f_fake)
    n_img, n_skip_mask = s["n_img"], s["n_skip_mask"]
    n_skip_missing, all_cov = s["n_skip_missing"], s["cov"]

    cov_stats = None
    if region != "all":
        cov = np.concatenate(all_cov)
        below = int((cov < args.min_coverage).sum())
        frac_below = below / float(cov.size)
        cov_stats = {
            "median_clip_coverage": round(float(np.median(cov)), 5),
            "mean_clip_coverage": round(float(cov.mean()), 5),
            "boxes_below_min_coverage": below,
            "frac_boxes_below_min_coverage": round(frac_below, 4),
        }
        if frac_below > 0.5:
            print(f"\n[WARNING] region={region}: {frac_below:.0%} of crops fall below "
                  f"min_coverage={args.min_coverage} (median coverage "
                  f"{np.median(cov):.5f}). These crops are mostly UNCLIPPED "
                  f"content, so this FID does not measure hallucination quality. "
                  f"This split has too little {region} clipping to support a "
                  f"region-restricted FID -- pick a split whose stops actually "
                  f"clip that end (e.g. shadow: hdrplus_linrgb_full_-12_-{{1,2,4}}_0_0_d4; "
                  f"highlight: any of the FiveK/HDR+ splits with a nonzero "
                  f"highlight clip).", file=sys.stderr)

    return {
        "fid": round(fid, 4),
        "coverage_diagnostics": cov_stats,
        "encoding": encoding,
        "region": region,
        "n_images": n_img,
        "n_patches": int(len(f_fake)),
        "reference": reference,
        "n_reference_patches": int(len(f_real)),
        "reference_source": (getattr(args, "reference_source", "all")
                             if reference == "domain" else None),
        "reference_family": (reference_family(args.data_dir)
                             if reference == "domain" else None),
        "crops_per_image": args.crops_per_image,
        "crop_size": args.crop_size,
        "seed": args.seed,
        "min_coverage": args.min_coverage if region != "all" else None,
        "mask_eps": round(args._mask_eps, 6) if region != "all" else None,
        "mask_source": (getattr(args, "mask_source", "recompute")
                        if region != "all" else None),
        "images_skipped_no_clipping": n_skip_mask,
        "images_skipped_missing": n_skip_missing,
        "feature_extractor": extractor.kind,
        "arm": "inverse_rescale_baseline" if args.baseline else "model",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True,
                    help="preprocessed split, e.g. data/fivek_linrgb_-12_-8_-4_0_d4")
    ap.add_argument("--file_list", default="HDRPlus_test.txt")
    ap.add_argument("--pred_dir", required=True,
                    help="dir with <name>_generated_linear.tiff")
    ap.add_argument("--crops_per_image", type=int, default=60)
    ap.add_argument("--crop_size", type=int, default=128)
    ap.add_argument("--reference_crops", type=int, default=20000,
                    help="number of unpaired reference crops")
    ap.add_argument("--reference_seed", type=int, default=99991,
                    help="seed for the reference crop set; keep it fixed across "
                         "--seed values so every arm meets the same reference")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None,
                    help="output yaml (default: fid.yaml beside the run's metric.yaml)")
    args = ap.parse_args()

    args.encoding = "reinhard"
    args.region = "all"
    args.reference = "domain"
    args.reference_source = "all"
    args.reference_list = None
    args.baseline = False
    args.mask_source = "recompute"
    args.min_coverage = 0.02   # argparse default of the full version; unused

    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(RD, args.data_dir)
    args._mask_eps = default_mask_eps(args.data_dir)
    if not args.pred_dir:
        raise SystemExit("[error] --pred_dir required")
    if not os.path.isabs(args.pred_dir):
        args.pred_dir = os.path.join(RD, args.pred_dir)
    if not os.path.isdir(args.pred_dir):
        raise SystemExit(f"[error] --pred_dir not found: {args.pred_dir}")

    pairs = read_pairs(args.data_dir, args.file_list)
    print(f"{len(pairs)} pairs from {args.file_list}")

    extractor = InceptionFeatures(args.device)
    print(f"feature extractor: {extractor.kind}")

    regions = [r.strip() for r in args.region.split(",") if r.strip()]
    results = {}
    for region, res in run_regions(args, pairs, regions, extractor).items():
        key = f"fid_{region}_{res['encoding']}"
        results[key] = res
        if res.get("fid") is None:
            print(f"\n=== {key} = SKIPPED ({res.get('skipped', 'no data')}) ===")
        else:
            print(f"\n=== {key} = {res['fid']:.3f}  "
                  f"({res['n_patches']} patches / {res['n_images']} images) ===")

    out = args.out
    if out is None:
        out = os.path.join(_run_dir(args.pred_dir), "fid.yaml")
    elif not os.path.isabs(out):
        out = os.path.join(RD, out)

    import yaml

    prev = {}
    if os.path.exists(out):
        with open(out) as fh:
            prev = yaml.safe_load(fh) or {}
    prev.update(results)
    with open(out, "w") as fh:
        yaml.dump(prev, fh, sort_keys=True, default_flow_style=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()