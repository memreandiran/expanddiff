# Checkpoints

⚠ These weights are licensed **CC BY-NC 4.0** — non-commercial use only.
See [`LICENSE-WEIGHTS.md`](LICENSE-WEIGHTS.md). The code in this repository
is Apache 2.0 and carries no such restriction.

Six checkpoints, one per arm reported in the paper and the supplementary
material. Each is ~96 MB and holds the model weights, the run's full Hydra
config and its step count. Optimizer state has been removed, so these are for
inference and not for resuming training; every weight tensor is bit-identical to
the checkpoint the reported numbers were produced from.

They are **not in the repository** — GitHub rejects files this large in a
checkout. Fetch them with:

    bash checkpoints/download.sh            # all six
    bash checkpoints/download.sh p b        # just ExpandDiff-P and -B

or download them by hand from the repository's Releases page into this folder.

## Which one to use

**ExpandDiff-P** unless you have a reason to pick another: it is the main
method, trained on inputs clipped at randomly sampled percentiles, and it is the
one to use on images clipped at both ends. **ExpandDiff-B** is trained on a
narrower, benchmark-like degradation — highlight clipping only, through a
randomised camera response — and is the stronger arm on inputs of that kind.
**ExpandDiff-D** is an earlier checkpoint trained on different data with a
linear target; it is included because the supplementary material reports it. The
three `ablation_*` files exist to reproduce the target-and-head ablation and are
not meant for general use.

Every checkpoint here is used the same way — over a folder of images:

    python training/inference_custom.py --checkpoint checkpoints/<file> \
      --input_dir <in> --output_dir <out> --linear_model --target_encoding <enc>

or over a preprocessed test split, scored against its targets:

    python training/sample.py checkpoint_path=checkpoints/<file> \
      general.is_linear=true general.target_encoding=<enc> ...

`<enc>` has to match the `target` column below: `pu21` for
`expanddiff_p_150k`, `expanddiff_b_75k` and `ablation_pu21_unbounded_150k`,
`none` for `expanddiff_d_150k`, `ablation_linear_target_150k` and
`ablation_linear_unbounded_150k`. A mismatch is not an error, it just produces a
wrongly-toned reconstruction, so the script warns when it disagrees with the
checkpoint. The `tanh head` column needs no flag: it is read from the
checkpoint.

| file | paper arm | steps | sha256 |
|---|---|---|---|
| `expanddiff_p_150k.ckpt` | **ExpandDiff-P** (main method) | 150000 | `d4fc4cfc757c2c01c3f01eba09fd32fa5b39584333075551cec6f2f815323bd3` |
| `expanddiff_b_75k.ckpt` | **ExpandDiff-B** | 75000 | `78607110e0bec46112daa5a6bc5327854bbb1df0b712cb0be761c4928b1700e1` |
| `expanddiff_d_150k.ckpt` | **ExpandDiff-D** (supplementary) | 150000 | `cf1dbc4075ea20ef85a036ed9e3c1df324e501bc95420efae62776246ae4e6c4` |
| `ablation_pu21_unbounded_150k.ckpt` | ablation, "PU21, no tanh" | 150000 | `a092539fd367384a94bf0c552c578ec8a8e63d4db360a5ce159c37e4000c8244` |
| `ablation_linear_target_150k.ckpt` | ablation, "linear, tanh" | 150000 | `76b514b81d733f729701e8f8e91524ccb3920accffeba5ccd01827fafc790da0` |
| `ablation_linear_unbounded_150k.ckpt` | ablation, "linear, no tanh" | 150000 | `a4c78e7323938e223243cdc3f7d56296ad899778870e091e838e20ba4d34d564` |

Verify with `sha256sum -c SHA256SUMS.txt`, or read a step count back directly:

    python -c "import torch;print(torch.load('expanddiff_p_150k.ckpt',map_location='cpu',weights_only=False)['global_step'])"

`weights_only=False` is required: Lightning stores an OmegaConf `DictConfig` in
the checkpoint, and torch >= 2.6 refuses to unpickle it by default, with an
`UnpicklingError` that reads like file corruption.

## Exactly what each checkpoint is

Every field below was read out of that run's own training log or its stored
config, not transcribed from notes. All six share `seed 0`, `is_linear true`,
`lr_scheduler cosine`, `L1 + L2` weights 1.0/1.0 (no log-L1 except D), and
DDIM-24 sampling.

| | data | target | tanh head | batch | lr | steps |
|---|---|---|---|---|---|---|
| **P** | `scenehdr_pct3_512` | PU21 | yes | 32 | 2e-4 | 150k |
| **B** | `scenehdr_c95_512` | PU21 | yes | 32 | **1e-4** | 75k |
| **D** | `hdrplus_linrgb_full_-12_-6_-4_0_d4` | linear (+ log-L1 1.0) | yes | 8 | 1e-4 | 150k |
| PU21, no tanh | `scenehdr_pct3_512` | PU21 | **no** | 32 | 2e-4 | 150k |
| linear, tanh | `scenehdr_pct3_512` | linear | yes | 32 | 2e-4 | 150k |
| linear, no tanh | `scenehdr_pct3_512` | linear | **no** | 32 | 2e-4 | 150k |

Note that **P and B differ in learning rate as well as in training degradation**
(2e-4 vs 1e-4). The three ablation rows are matched to P on every field except
the one each varies.
