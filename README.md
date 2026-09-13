# ExpandDiff

Code and checkpoints for **ExpandDiff: Dynamic Range Expanding Diffusion for
Single-Image HDR Reconstruction**. The model reconstructs crushed shadows and
blown highlights from a single LDR image. The paper explains how it works; this
file explains how to run it.

Project page: <https://memreandiran.github.io/expanddiff/>

    training/        train.py, sample.py, inference_custom.py, Hydra configs
    preprocessing/   build training splits and evaluation splits
    metrics/         everything used to produce the reported numbers
    rawdiffusion/    the model, diffusion, datasets and metrics
    scripts/         environment check, one-command evaluation
    checkpoints/     the six released checkpoints (weights fetched separately)

## Install

    conda create -n expanddiff python=3.11 && conda activate expanddiff
    pip install -r requirements.txt
    bash scripts/check_environment.sh

The check prints one line per metric family. Training, sampling, alignment and
PU21-PSNR need only `requirements.txt`. The rest need software we cannot
redistribute:

| for | install | point at it with |
|---|---|---|
| PU21-VSI, +CRF columns | <https://github.com/gfxdisp/pu21> | `PU21_M` |
| PU21-VSI, +CRF, HDR-VDP-3 | Octave | `OCT_BIN`, `OCTAVE_HOME` |
| HDR-VDP-3 | 3.0.7, <https://hdrvdp.sourceforge.net> | `VDP_ROOT` |
| PU21-PIQE | `pip install pyiqa` | — |
| FID-R | `pip install torch-fidelity` | — |

Reading EXR files needs OpenCV's OpenEXR backend:
`export OPENCV_IO_ENABLE_OPENEXR=1`.

## Reconstruct an image

    bash checkpoints/download.sh p

    python training/inference_custom.py \
      --checkpoint checkpoints/expanddiff_p_150k.ckpt \
      --input_dir  <folder of png/jpg> \
      --output_dir <folder for results> \
      --linear_model --target_encoding pu21

You get `<name>_generated_linear.tiff`, float32 linear radiance where `1.0`
means 1000 cd/m², and `<name>_generated_srgb.png` to look at. Add
`--save_guidance` to also write what the model was fed.

`--target_encoding` must match the checkpoint: `pu21` for ExpandDiff-P,
ExpandDiff-B and the PU21 ablation, `none` for ExpandDiff-D and the two linear
ablations. `checkpoints/MANIFEST.md` lists it per file and says which checkpoint
to use.

## Reproduce the evaluation

Download SI-HDR from
<https://www.cl.cam.ac.uk/research/rainbow/projects/sihdr_benchmark/>, then:

    # build the benchmark condition
    # (or sihdr_pctclip_preprocess.py for the two-sided clipping condition)
    python preprocessing/sihdr_preprocess.py --dataset_root <SI-HDR> \
        --output_dir data/c95 --clip_level clip_95

    # sample, align and score in one command
    bash scripts/evaluate_checkpoint.sh checkpoints/expanddiff_b_75k.ckpt \
        data/c95 SIHDR_test.txt pu21 out/

`<SI-HDR>` is the directory holding `reference/` and `input/`; the released
archives unpack one level deeper.

`scripts/evaluate_checkpoint.sh` builds the 8-bit input every method is scored on,
samples, fits the per-image brightness gain, and runs each metric family it has
the dependencies for. To run those steps yourself, see
[`metrics/README.md`](metrics/README.md).

## Train

Build a training split from scene-referred HDR — Poly Haven
(<https://polyhaven.com/hdris>), the Laval photometric sample
(<http://hdrdb.com/>), HDR-Real (<https://alex04072000.github.io/SingleHDR/>)
and Fairchild (<http://markfairchild.org/HDR.html>) — then train:

    DATASETS_DIR=<dir> bash preprocessing/run_scenehdr_pct_build.sh   # ExpandDiff-P
    DATASETS_DIR=<dir> bash preprocessing/run_scenehdr_c95_build.sh   # ExpandDiff-B

    python training/train.py general.is_linear=true general.target_encoding=pu21 \
        general.weight_logl1=0.0 general.weight_pul1=0.0 \
        dataset.train.data_dir=data/scenehdr_pct_512 \
        dataset.val.data_dir=data/scenehdr_pct_512 \
        dataset.train.file_list=SceneHDR_train.txt \
        dataset.train.batch_size=32 general.max_steps=150000 \
        general.lr=2e-4 general.lr_scheduler=cosine general.seed=0 \
        general.suffix=my_run general.check_val_every_n_epoch=10

`general.lr` is not part of the experiment directory name, so always pass it
explicitly. `checkpoints/MANIFEST.md` gives every setting of all six released
runs. To sample from a checkpoint, pass `checkpoint_path=<file>.ckpt` to
`training/sample.py`.

## Every step on its own

`scripts/evaluate_checkpoint.sh` bundles steps 2b to 5. Run them individually like this:

    # 0. build the evaluation split from the SI-HDR benchmark
    #    <SI-HDR> is the directory holding reference/ and input/; the released
    #    archives unpack one level deeper, as reference/sihdr/reference
    python preprocessing/sihdr_preprocess.py --dataset_root <SI-HDR> \
        --output_dir <test split> --clip_level clip_95
    #    (or sihdr_pctclip_preprocess.py for the doubly-clipped condition)

    # 1. preprocess: HDR sources -> training pairs
    python preprocessing/scenehdr_preprocess.py --dataset_root <hdr dir> [--panorama] \
        --output_dir data/<split> --size 512 512 --degradation percentile \
        --clip_pct_low 0 10 --clip_pct_high 0 30 --clip_repeats 3 [--append]

    # 2. train
    python training/train.py general.is_linear=true general.target_encoding=pu21 \
        general.weight_logl1=0.0 general.weight_pul1=0.0 \
        dataset.train.data_dir=data/<split> dataset.val.data_dir=data/<split> \
        dataset.train.file_list=<prefix>.txt dataset.train.batch_size=32 \
        general.max_steps=150000 general.lr=2e-4 general.lr_scheduler=cosine \
        general.seed=0 general.suffix=<name> general.check_val_every_n_epoch=10

    # 2b. put every method on the same 8-bit input before evaluating
    python preprocessing/make_q8_split.py --src <test split> --dst <test split>_q8 \
        --file_list <prefix>.txt

    # 3a. sample a test split  (sample from the _q8 twin, score against the original)
    python training/sample.py checkpoint_path=checkpoints/<file>.ckpt \
        general.is_linear=true general.target_encoding=pu21 \
        general.weight_logl1=0.0 \
        dataset.train.data_dir=data/<split> dataset.train.file_list=<prefix>.txt \
        dataset.train.batch_size=32 \
        dataset.val.data_dir=<test split>_q8 dataset.val.file_list=<prefix>.txt \
        general.max_steps=150000 general.lr_scheduler=cosine \
        general.suffix=<name> general.check_val_every_n_epoch=10

    # 3b. or run a checkpoint over any image files
    python training/inference_custom.py --checkpoint checkpoints/<file>.ckpt \
        --input_dir <in> --output_dir <out> \
        --linear_model --target_encoding pu21 --save_guidance

    # 4. align the predictions
    python metrics/align.py --raw_dir <predictions> --split_dir <test split> \
        --align scale --out_dir <arm>_scale/pred

    # 5. score
    python metrics/compute_ref_metrics.py --device cpu --linear \
        --data_dir <test split> --file_list <prefix>.txt \
        --pred_dir <arm>_scale/pred --out gain.yaml
    python metrics/vsi_ref_cells.py --split_dir <test split> \
        --condition <condition> --arms <arm>_scale --out_dir vsi/
    python metrics/crf_ref2_cells.py --split_dir <test split> \
        --condition <condition> --arms <arm>_scale --out_dir crf/
    python metrics/compute_fid.py --device cpu \
        --data_dir <test split> --file_list <prefix>.txt \
        --pred_dir <arm>_scale/pred --out fid.yaml
    VDP_ROOT=<hdrvdp-3.0.7> DIAG_IN=24 RES_W=1920 RES_H=1080 DIST_M=1.0 \
        python metrics/hdrvdp3_bridge.py --data_dir <test split> \
        --pred_dir <arm>_scale/pred --file_list <prefix>.txt --out vdp3.yaml
    python metrics/piqe_ref_cells.py --split_dir <test split> \
        --condition <condition> --arms <arm>_scale --out_dir piqe/

    # 6. HDR-VDP-3 on the corrected basis: write corrected tiffs, then score them
    python metrics/crf_apply_dir_ref.py --data_dir <test split> \
        --pred_dir <arm>_scale/pred --file_list <prefix>.txt \
        --out_dir <arm>_scale_crf/pred
    VDP_ROOT=<hdrvdp-3.0.7> DIAG_IN=24 RES_W=1920 RES_H=1080 DIST_M=1.0 \
        python metrics/hdrvdp3_bridge.py --data_dir <test split> \
        --pred_dir <arm>_scale_crf/pred --file_list <prefix>.txt --out vdp3crf.yaml

    # 7. compare two methods
    python metrics/paired_test.py --curve pu21 --a <arm_a>_scale/pred \
        --b <arm_b>_scale/pred --target_dir <test split>/<prefix>_target

## Licence

**Code:** Apache 2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
**Checkpoints:** CC BY-NC 4.0, non-commercial use only; see
[`checkpoints/LICENSE-WEIGHTS.md`](checkpoints/LICENSE-WEIGHTS.md).

The backbone is
adapted from RAW-Diffusion (Reinders et al.), the guidance encoder follows EDSR
(Lim et al.), the conditioning follows SPADE (Park et al.), and PU21 is Mantiuk
and Azimi's.

## Acknowledgement

This project builds on code from the following repositories. We thank the
authors for publishing their work:

* [RAW-Diffusion](https://github.com/SonyResearch/RAW-Diffusion) — the
  conditional diffusion backbone we adapt
* [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch) — the guidance
  encoder
* [SPADE](https://github.com/NVlabs/SPADE) — the spatially-adaptive
  normalisation used for conditioning
* [pu21](https://github.com/gfxdisp/pu21) — the PU21 encoding, the reference
  20-parameter correction, and VSI (Zhang et al.)
* [HDR-VDP-3](https://hdrvdp.sourceforge.net) — the perceptual quality metric
* [torch-fidelity](https://github.com/toshas/torch-fidelity) — FID and its
  canonical Inception weights
* [piq](https://github.com/photosynthesis-team/piq) and
  [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) — SSIM/MS-SSIM and
  PIQE
* [LPIPS](https://github.com/richzhang/PerceptualSimilarity)

We evaluate against ExpandNet, MaskHDR, LEDiff, DITM and Refusion-HDR by running
the weights and code their authors released, and on the
[SI-HDR benchmark](https://www.cl.cam.ac.uk/research/rainbow/projects/sihdr_benchmark/)
of Hanji et al. Training uses HDR from Poly Haven, the Laval photometric
sample, HDR-Real and the Fairchild HDR Photographic Survey.

## Citation

*BibTeX to follow.*
