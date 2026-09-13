"""Run ExpandDiff over a preprocessed test split and score the result.

Use this when you have a split in the layout the preprocessors write (an index
file plus `*_target` and `*_guidance` folders of float32 .npy). To run over any
image files, with no ground truth needed, use `inference_custom.py` instead.

    python training/sample.py general.is_linear=true general.target_encoding=pu21 \
        general.weight_logl1=0.0 \
        dataset.train.data_dir=data/scenehdr_pct_512 \
        dataset.train.file_list=SceneHDR_train.txt dataset.train.batch_size=32 \
        dataset.val.data_dir=<test split> dataset.val.file_list=<list>.txt \
        general.max_steps=150000 general.lr_scheduler=cosine \
        general.suffix=scenehdr_pct3 general.check_val_every_n_epoch=10

WHICH CHECKPOINT GETS LOADED. Either the one you name, or one resolved from the
config:

  checkpoint_path=checkpoints/expanddiff_p_150k.ckpt   <- loads that file

With `checkpoint_path` set, nothing else has to line up: the experiment
directory is still derived from the config and still receives the outputs, but
the checkpoint comes from where you said. This is how a released checkpoint is
used.

Left unset, the file loaded is `<experiment>/checkpoints/<checkpoint_name>`
(`last.ckpt` by default), where `<experiment>` is rebuilt from the training
config — data dir, batch size, step budget, loss weights, target encoding,
suffix. That is convenient right after training, when your own run wrote that
directory, and awkward otherwise.

WHAT YOU GET, under `<experiment>/inference_sampling/<val split name>/`:

    visualizations/<name>_generated_linear.tiff   the prediction: float32 linear
                                                  radiance, 1.0 = 1000 cd/m^2
    visualizations/<name>_guide_linear.tiff       the LDR input it was given
    visualizations/<name>_{gt,generated,guide}_srgb.png   8-bit previews, for
                                                  looking at rather than scoring
    metric.yaml                                   aggregate scores over the split
    per_image_stats.txt                           the same, per image

⚠ The scores written here are computed on the RAW prediction, before the
brightness alignment described in the paper, so they are several dB below the
published numbers and are not comparable to them. They are a sanity check while
sampling. To reproduce a reported number, take the tiffs, run `metrics/align.py`
over them, then score with the tools in `metrics/`; `metrics/README.md` walks
through it end to end.

The predicted tiff is scene-linear: multiply by 1000 for cd/m^2, or apply a tone
map before viewing. Predictions are decoded out of PU21 before being written, so
every downstream tool reads plain linear radiance.
"""
# Make the repository root importable no matter where this is run from:
# Python puts the SCRIPT's directory on sys.path, not the working directory, so
# `python training/train.py` would otherwise fail to find `rawdiffusion`.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os

import h5py
import hydra
import numpy as np
import tifffile
import torch
import torchvision as tv
import yaml
from omegaconf import DictConfig, OmegaConf
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from rawdiffusion.evaluation.collection import CollectionMetric
from rawdiffusion.evaluation.metrics import (
    CosineDistanceMetric,
    LPIPSMetric,
    MSEMetric,
    PearsonMetric,
    PSNRMetric,
    PUMSSSIMMetric,
    MuLawPSNRMetric,
    MuLawSSIMMetric,
    PU21PSNRMetric,
    PU21SSIMMetric,
    PUPSNRMetric,
    PUSSIMMetric,
    SSIMMetric,
    pu_encode,
)
from rawdiffusion.datasets.dataset_factory import create_dataset
from rawdiffusion.gaussian_diffusion_factory import create_gaussian_diffusion
from rawdiffusion.utils import decode_target, get_output_path, linear_to_srgb
from train import RAWDiffusionModule
from rawdiffusion.config import mod_config
from rawdiffusion.utils import create_folder_for_file


def save_float_tiff(tensor_chw, path):
    """Save a (C, H, W) torch tensor in [0, 1] as a float32 TIFF
    with no quantization. Used for linear-RGB guidance and generated
    images so downstream HDR consumers get unrounded values."""
    arr = tensor_chw.detach().cpu().to(torch.float32).numpy()
    arr = np.transpose(arr, (1, 2, 0))  # (H, W, C)
    tifffile.imwrite(path, arr, photometric="rgb")


def load_clipmasks_for(rel_path, data_dir):
    """Look for the precomputed (8, H, W) uint8 clipmasks file the
    preprocessing pipeline writes next to each test guidance .npy.
    Returns 8 boolean (H, W) torch tensors in the order
    (orig_shadow_any, orig_shadow_all, orig_highlight_any,
     orig_highlight_all, art_shadow_any, art_shadow_all,
     art_highlight_any, art_highlight_all),
    or None if the file is missing."""
    if not rel_path or not rel_path.endswith(".npy"):
        return None
    clipmasks_path = os.path.join(
        data_dir, rel_path[:-4] + "_clipmasks.npy"
    )
    if not os.path.isfile(clipmasks_path):
        return None
    masks = np.load(clipmasks_path).astype(bool)  # (8, H, W)
    masks_t = torch.from_numpy(masks)
    return tuple(masks_t[i] for i in range(masks_t.shape[0]))


def compute_clipmasks_fallback(target_disp_j, guidance_data_j, eps=0.0):
    """Same semantics as the preprocessing's compute_clipmasks(): 8
    boolean (H, W) tensors splitting orig/art and shadow/highlight at
    both any-channel and all-channel granularity. Used when a
    precomputed clipmasks file is missing."""
    t = target_disp_j.cpu()
    g = guidance_data_j.cpu()
    orig_low_pc = t <= eps
    orig_high_pc = t >= 1.0 - eps
    art_low_pc = (g <= eps) & ~orig_low_pc
    art_high_pc = (g >= 1.0 - eps) & ~orig_high_pc
    return (
        orig_low_pc.any(0),  orig_low_pc.all(0),
        orig_high_pc.any(0), orig_high_pc.all(0),
        art_low_pc.any(0),   art_low_pc.all(0),
        art_high_pc.any(0),  art_high_pc.all(0),
    )


def paint_clip_overlay(img, art_shadow, art_highlight, orig_shadow, orig_highlight):
    """Paint clipped pixels:
        artificial shadow    -> blue   (0, 0, 1)
        artificial highlight -> red    (1, 0, 0)
        original  shadow     -> cyan   (0, 1, 1)
        original  highlight  -> yellow (1, 1, 0)
    Artificial overlays win where original and artificial overlap, so
    the visualisation always shows what we crushed."""
    out = img.clone()

    # Original (already in source) -- pastel colours, drawn first.
    only_orig_shadow = orig_shadow & ~art_shadow & ~art_highlight
    out[0, only_orig_shadow] = 0.0
    out[1, only_orig_shadow] = 1.0
    out[2, only_orig_shadow] = 1.0

    only_orig_highlight = orig_highlight & ~art_shadow & ~art_highlight
    out[0, only_orig_highlight] = 1.0
    out[1, only_orig_highlight] = 1.0
    out[2, only_orig_highlight] = 0.0

    # Artificial -- saturated, drawn second so they win on overlap.
    out[0, art_shadow] = 0.0
    out[1, art_shadow] = 0.0
    out[2, art_shadow] = 1.0

    out[0, art_highlight] = 1.0
    out[1, art_highlight] = 0.0
    out[2, art_highlight] = 0.0
    return out


def get_val_output_name(cfg):
    output_name = cfg.dataset.val.data_dir
    output_name = os.path.basename(os.path.normpath(output_name))

    file_list_name = os.path.splitext(os.path.basename(cfg.dataset.val.file_list))[0]
    output_name += "_" + file_list_name

    if cfg.diffusion_val.timestep_respacing:
        output_name += "_" + cfg.diffusion_val.timestep_respacing

    if cfg.get("sample_center_crop", False):
        output_name += f"_cc{cfg.general.image_size}"

    return output_name


@hydra.main(
    version_base="1.3", config_path="configs", config_name="rawdiffusion_sample"
)
def main(cfg: DictConfig) -> None:
    mod_config(cfg)
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    experiment_folder = get_output_path(cfg)
    print(f"experiment_folder: {experiment_folder}")

    # `checkpoint_path=<file>` loads that checkpoint directly, which is how a
    # released checkpoint is used: the experiment directory is still derived
    # from the config and still receives the outputs, but nothing has to be
    # copied into it first. Left unset, the checkpoint is resolved from that
    # directory exactly as before.
    checkpoint_path = cfg.get("checkpoint_path", None) or os.path.join(
        experiment_folder, "checkpoints", cfg.checkpoint_name
    )
    print(f"checkpoint_path: {checkpoint_path}")
    if not os.path.isfile(checkpoint_path):
        raise SystemExit(
            f"no checkpoint at {checkpoint_path}\n"
            "Pass checkpoint_path=<file> to load a released checkpoint directly, "
            "or check that the training arguments match the run you meant."
        )

    output_name = get_val_output_name(cfg)
    print(f"inference output name: {output_name}")

    # Architecture settings come from the checkpoint itself, so a checkpoint
    # trained without the bounded head is not silently rebuilt with one. The
    # config default (out_tanh: True) would otherwise win, and the only symptom
    # is a wrong reconstruction: tanh applied to a model that never had it.
    _ck = torch.load(checkpoint_path, map_location="cpu",
                     weights_only=False).get("hyper_parameters", {})
    _ck_model = _ck.get("model", {}) if isinstance(_ck, dict) else {}
    if "out_tanh" in _ck_model and _ck_model["out_tanh"] != cfg.model.out_tanh:
        print(f"checkpoint was trained with out_tanh={_ck_model['out_tanh']}; "
              f"using that instead of the config default {cfg.model.out_tanh}")
        cfg.model.out_tanh = _ck_model["out_tanh"]
    _ck_enc = (_ck.get("general", {}) or {}).get("target_encoding", "none") \
        if isinstance(_ck, dict) else "none"
    if _ck_enc != cfg.general.get("target_encoding", "none"):
        print(f"WARNING: checkpoint was trained with target_encoding={_ck_enc}, "
              f"but general.target_encoding="
              f"{cfg.general.get('target_encoding', 'none')} was requested. The "
              f"prediction will be decoded with the wrong curve.")

    # `checkpoint_path` is consumed above; its name collides with
    # load_from_checkpoint's own first argument, so it must not be splatted.
    mod_kwargs = {k: v for k, v in cfg.items() if k != "checkpoint_path"}
    raw_module = RAWDiffusionModule.load_from_checkpoint(
        checkpoint_path, experiment_folder=experiment_folder, **mod_kwargs
    )

    sample_center_crop = bool(cfg.get("sample_center_crop", False))
    # ⚠ target_encoding MUST be passed here. Without it create_dataset defaults
    # to "none", the loader returns a LINEAR target, and the decode_target call
    # below then decodes a target that was never encoded -- a double decode that
    # collapsed the GT toward black on every PU21 run. Verified 2026-08-28:
    # _gt_srgb.png matched linear_to_srgb(pu21_decode(target)) at mean 19.20
    # against an observed 19.10, where the correct render is mean 72.30.
    # train.py has always passed it to BOTH datasets (train.py:433-442), so
    # training and val metrics were never affected -- only sample.py's previews
    # and its own (already-discarded, unaligned) metric.yaml.
    # Predictions are UNCHANGED by this fix: `sample` comes from the model, is
    # always encoded, and is decoded exactly once either way.
    data_val = create_dataset(
        **cfg.dataset.val,
        transform=sample_center_crop,
        patch_size=cfg.general.image_size,
        seed=cfg.general.seed,
        target_encoding=cfg.general.get("target_encoding", "none"),
    )
    # The data below is moved to the GPU explicitly, so the module has to be
    # too. A checkpoint written during GPU training happens to restore onto
    # the GPU by itself, which is why this was never needed before; a
    # checkpoint saved from CPU (every released one) restores onto the CPU
    # and the first convolution then fails on mismatched tensor types.
    raw_module.eval()
    if torch.cuda.is_available():
        raw_module.cuda()

    image_path = os.path.join(
        experiment_folder, "inference_sampling", output_name, "visualizations"
    )
    os.makedirs(image_path, exist_ok=True)
    inference_output_path = os.path.join(
        experiment_folder,
        "inference_sampling",
        output_name,
    )
    os.makedirs(inference_output_path, exist_ok=True)

    config_path = os.path.join(
        experiment_folder, "inference_sampling", output_name, "config.yaml"
    )
    metric_path = os.path.join(
        experiment_folder, "inference_sampling", output_name, "metric.yaml"
    )

    metrics_dict = {
        "mse": MSEMetric(),
        "psnr": PSNRMetric(),
        "ssim": SSIMMetric(),
        "pearson": PearsonMetric(),
        "lpips": LPIPSMetric(input_is_linear=cfg.general.is_linear),
        "cosine_distance": CosineDistanceMetric(),
    }
    # PU-encoded PSNR/SSIM are calibrated for *linear* luminance, so only
    # add them when the inputs are linear-RGB. Applying PU to sRGB-encoded
    # values would double-warp the tone curve.
    if cfg.general.is_linear:
        metrics_dict["pu_psnr"] = PUPSNRMetric()
        # PU21 (Mantiuk 2021) -- a different curve from the Aydin PU above,
        # and the one AIM 2025 ranks on. Reported alongside, not instead.
        metrics_dict["pu21_psnr"] = PU21PSNRMetric()
        metrics_dict["pu21_ssim"] = PU21SSIMMetric()
        # reference convention (gfxdisp/pu21): `pu21_psnr_ref` is the field the
        # paper reports and the only one comparable to a published PU21-PSNR.
        # The normalised `pu21_psnr` above is a different convention and differs
        # from it by a constant, so never quote the two interchangeably.
        metrics_dict["pu21_psnr_ref"] = PU21PSNRMetric(convention="reference")
        metrics_dict["pu21_ssim_ref"] = PU21SSIMMetric(convention="reference")
        # mu-law: the Kalantari-lineage headline, and what ExpoCM reports
        metrics_dict["mu_psnr"] = MuLawPSNRMetric()
        metrics_dict["mu_ssim"] = MuLawSSIMMetric()
        metrics_dict["pu_ssim"] = PUSSIMMetric()
        metrics_dict["pu_ms_ssim"] = PUMSSSIMMetric()
    metrics_sampling = CollectionMetric(metrics_dict)

    diffusion = create_gaussian_diffusion(**cfg.diffusion_val)

    use_ddim = "ddim" in cfg.diffusion_val.timestep_respacing
    clip_denoised = True

    save_visualization_interval = cfg.save_visualization_interval
    save_timesteps = cfg.save_timesteps
    save_pred = cfg.save_pred
    save_tar = cfg.save_tar

    save_as_hdf5 = cfg.save_as_hdf5

    model = raw_module.model

    all_grid_psnrs = []
    all_grid_ssims = []

    per_image_stats_path = os.path.join(
        experiment_folder, "inference_sampling", output_name, "per_image_stats.txt"
    )
    stats_file = open(per_image_stats_path, "w")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("PSNR: {task.fields[psnr_rgb]}"),
        TextColumn("SSIM: {task.fields[ssim_rgb]}"),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    ) as progress:
        task_total_id = progress.add_task(
            "[red]Total Dataset", total=len(data_val), psnr_rgb="", ssim_rgb=""
        )
        task_batch_id = progress.add_task(
            "[green]Batch", total=100, psnr_rgb="", ssim_rgb=""
        )
        task_total = progress._tasks[task_total_id]

        print("sampling...")
        num_samples = 0

        for batch in data_val:
            target_data = batch["target_data"].cuda()
            guidance_data = batch["guidance_data"].cuda()

            # Pad to multiple of 64 for U-Net compatibility
            _, _, h_orig, w_orig = target_data.shape
            pad_h = (64 - h_orig % 64) % 64
            pad_w = (64 - w_orig % 64) % 64
            if pad_h > 0 or pad_w > 0:
                target_data = torch.nn.functional.pad(target_data, (0, pad_w, 0, pad_h), mode='reflect')
                guidance_data = torch.nn.functional.pad(guidance_data, (0, pad_w, 0, pad_h), mode='reflect')

            guidance_input = raw_module.preprocess_guidance(guidance_data)
            guidance_input = {k: v.cuda() for k, v in guidance_input.items()}

            progress.reset(task_batch_id, total=diffusion.num_timesteps)

            ts = diffusion.num_timesteps - 1
            noise = torch.randn_like(target_data)

            indices = list(range(ts))[::-1]
            sample_fn_progressive = (
                diffusion.p_sample_loop_progressive
                if not use_ddim
                else diffusion.ddim_sample_loop_progressive
            )

            vis_step = max(1, diffusion.num_timesteps // 8)
            samples = []

            with torch.inference_mode():
                for sample_dict in sample_fn_progressive(
                    model,
                    shape=(
                        guidance_data.shape[0],
                        3,
                        guidance_data.shape[2],
                        guidance_data.shape[3],
                    ),
                    noise=noise,
                    clip_denoised=clip_denoised,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=guidance_input,
                    device=None,
                    progress=False,
                    progress_fn=lambda: progress.advance(task_batch_id, advance=1),
                    indices=indices,
                ):
                    sample_t = sample_dict["t"]
                    sample = sample_dict["sample"]
                    if sample_t % vis_step == 0 or sample_t == 0:
                        samples.append(sample)

            # Crop back to original size
            if pad_h > 0 or pad_w > 0:
                sample = sample[:, :, :h_orig, :w_orig]
                target_data = target_data[:, :, :h_orig, :w_orig]
                samples = [s[:, :, :h_orig, :w_orig] for s in samples]
                noise = noise[:, :, :h_orig, :w_orig]
                guidance_data = guidance_data[:, :, :h_orig, :w_orig]

            sample = (sample + 1) / 2.0
            target_data = (target_data + 1) / 2.0
            samples = [(sample + 1) / 2.0 for sample in samples]
            noise = (noise + 1) / 2.0
            guidance_data = (guidance_data + 1) / 2.0

            # This is the step that turns a bounded tanh prediction back into
            # high dynamic range. Target and prediction are both encoded, so
            # both are decoded; guidance never was, so it is left alone.
            # Identity when target_encoding=none, i.e. for every existing run.
            enc = cfg.general.get("target_encoding", "none")
            sample = decode_target(sample, enc)
            target_data = decode_target(target_data, enc)
            samples = [decode_target(s, enc) for s in samples]

            metrics_sampling.update(target_data, sample)

            target_disp = target_data.clamp(0, 1)
            samples_disp = [s.clamp(0, 1) for s in samples]
            sample_disp = sample.clamp(0, 1)

            sample_np = sample.cpu().numpy()
            batch_np = target_data.cpu().numpy()

            for j in range(sample.shape[0]):
                rel_path = batch["path"][j]

                if os.path.isabs(rel_path):
                    rel_path = os.path.basename(rel_path)
                fn = os.path.splitext(rel_path)[0]

                if torch.isnan(sample[j]).any():
                    print("sample is nan. skipping")
                    continue

                # Load precomputed clipmasks written by the preprocessing
                # pipeline. Falls back to in-place computation if the file
                # is missing (older runs).
                #   shadow clipped       -> blue
                #   highlight clipped    -> red
                #   original (source)    -> pastel shade of the same hue
                g_j = guidance_data[j].cpu()
                loaded_masks = load_clipmasks_for(
                    batch["path"][j], cfg.dataset.val.data_dir
                )
                if loaded_masks is not None:
                    # Center-crop the on-disk full-res masks to match the
                    # cropped image tensor when sample_center_crop is on.
                    H_img, W_img = g_j.shape[-2], g_j.shape[-1]
                    H_msk, W_msk = loaded_masks[0].shape
                    if (H_msk, W_msk) != (H_img, W_img):
                        oy = (H_msk - H_img) // 2
                        ox = (W_msk - W_img) // 2
                        loaded_masks = tuple(
                            m[oy:oy + H_img, ox:ox + W_img] for m in loaded_masks
                        )
                    (orig_sh_any, orig_sh_all,
                     orig_hi_any, orig_hi_all,
                     art_sh_any,  art_sh_all,
                     art_hi_any,  art_hi_all) = loaded_masks
                else:
                    (orig_sh_any, orig_sh_all,
                     orig_hi_any, orig_hi_all,
                     art_sh_any,  art_sh_all,
                     art_hi_any,  art_hi_all) = compute_clipmasks_fallback(
                        target_disp[j], guidance_data[j]
                    )

                if cfg.general.is_linear:
                    gt_data_path = os.path.join(image_path, fn + "_gt_linear.png")
                    create_folder_for_file(gt_data_path)
                    tv.utils.save_image(target_disp[j],   gt_data_path)
                    # Linear guidance and generated images are saved as
                    # float32 TIFFs so downstream HDR operations are not
                    # forced to round-trip through 8-bit quantization.
                    save_float_tiff(sample_disp[j],   os.path.join(image_path, fn + "_generated_linear.tiff"))
                    save_float_tiff(guidance_data[j], os.path.join(image_path, fn + "_guide_linear.tiff"))
                    tv.utils.save_image(paint_clip_overlay(g_j, art_sh_any, art_hi_any, orig_sh_any, orig_hi_any), os.path.join(image_path, fn + "_guide_clipmask_any_linear.png"))
                    tv.utils.save_image(paint_clip_overlay(g_j, art_sh_all, art_hi_all, orig_sh_all, orig_hi_all), os.path.join(image_path, fn + "_guide_clipmask_all_linear.png"))

                    g_srgb = linear_to_srgb(g_j)
                    tv.utils.save_image(linear_to_srgb(target_disp[j]),   os.path.join(image_path, fn + "_gt_srgb.png"))
                    tv.utils.save_image(linear_to_srgb(sample_disp[j]),   os.path.join(image_path, fn + "_generated_srgb.png"))
                    tv.utils.save_image(g_srgb,                            os.path.join(image_path, fn + "_guide_srgb.png"))
                    tv.utils.save_image(paint_clip_overlay(g_srgb, art_sh_any, art_hi_any, orig_sh_any, orig_hi_any), os.path.join(image_path, fn + "_guide_clipmask_any_srgb.png"))
                    tv.utils.save_image(paint_clip_overlay(g_srgb, art_sh_all, art_hi_all, orig_sh_all, orig_hi_all), os.path.join(image_path, fn + "_guide_clipmask_all_srgb.png"))
                else:
                    gt_data_path = os.path.join(image_path, fn + "_gt.png")
                    create_folder_for_file(gt_data_path)
                    tv.utils.save_image(target_disp[j],   gt_data_path)
                    tv.utils.save_image(sample_disp[j],   os.path.join(image_path, fn + "_generated.png"))
                    tv.utils.save_image(guidance_data[j], os.path.join(image_path, fn + "_guide.png"))
                    tv.utils.save_image(paint_clip_overlay(g_j, art_sh_any, art_hi_any, orig_sh_any, orig_hi_any), os.path.join(image_path, fn + "_guide_clipmask_any.png"))
                    tv.utils.save_image(paint_clip_overlay(g_j, art_sh_all, art_hi_all, orig_sh_all, orig_hi_all), os.path.join(image_path, fn + "_guide_clipmask_all.png"))

                # Per-image channel stats and metrics to file
                ch_names = ["R", "G", "B"]
                # Per-image PSNR/SSIM
                from skimage.metrics import peak_signal_noise_ratio, structural_similarity
                gt_rgb_np = target_data[j:j+1].cpu().numpy()[0].transpose(1, 2, 0)
                pr_rgb_np = sample[j:j+1].cpu().numpy()[0].transpose(1, 2, 0)
                img_psnr = peak_signal_noise_ratio(gt_rgb_np, pr_rgb_np, data_range=1.0)
                img_ssim = structural_similarity(gt_rgb_np, pr_rgb_np, data_range=1.0, channel_axis=2)
                with torch.no_grad():
                    lpips_metric = metrics_sampling.metrics["lpips"]
                    img_lpips = lpips_metric.lpips(
                        lpips_metric._to_lpips_input(sample[j:j+1]),
                        lpips_metric._to_lpips_input(target_data[j:j+1]),
                    ).item()

                # HDR-aware per-image extras: PU-PSNR / PU-SSIM (linear only) and
                # cosine distance (always). PU metrics use the same Aydin 2008
                # log encoding as the aggregate; cosine distance is mean
                # (1 - cos(pred_rgb, gt_rgb)) over pixels.
                img_pu_psnr = None
                img_pu_ssim = None
                img_pu_ms_ssim = None
                if cfg.general.is_linear:
                    with torch.no_grad():
                        gt_pu = pu_encode(target_data[j:j+1])
                        pr_pu = pu_encode(sample[j:j+1])
                    gt_pu_np = gt_pu[0].cpu().numpy().transpose(1, 2, 0)
                    pr_pu_np = pr_pu[0].cpu().numpy().transpose(1, 2, 0)
                    img_pu_psnr = peak_signal_noise_ratio(gt_pu_np, pr_pu_np, data_range=1.0)
                    img_pu_ssim = structural_similarity(gt_pu_np, pr_pu_np, data_range=1.0, channel_axis=2)
                    # Per-image MS-SSIM via torchmetrics (skimage has no MS-SSIM).
                    # The aggregate metric object is reused so the moving-average
                    # state is unaffected.
                    with torch.no_grad():
                        img_pu_ms_ssim = float(
                            metrics_sampling.metrics["pu_ms_ssim"].ms_ssim(
                                gt_pu, pr_pu
                            ).item()
                        )
                # Cosine distance (per-pixel RGB-direction error, averaged).
                with torch.no_grad():
                    pj = sample[j].cpu()
                    tj = target_data[j].cpu()
                    eps = 1e-8
                    dot = (pj * tj).sum(dim=0)
                    pn = pj.pow(2).sum(dim=0).clamp(min=eps).sqrt()
                    tn = tj.pow(2).sum(dim=0).clamp(min=eps).sqrt()
                    img_cosdist = float((1.0 - dot / (pn * tn)).mean().item())

                # 3x3 grid patch-based evaluation (same as paper)
                h_img, w_img = gt_rgb_np.shape[:2]
                ph, pw = h_img // 3, w_img // 3
                patch_psnrs, patch_ssims = [], []
                for gi in range(3):
                    for gj in range(3):
                        gt_patch = gt_rgb_np[gi*ph:(gi+1)*ph, gj*pw:(gj+1)*pw]
                        pr_patch = pr_rgb_np[gi*ph:(gi+1)*ph, gj*pw:(gj+1)*pw]
                        patch_psnrs.append(peak_signal_noise_ratio(gt_patch, pr_patch, data_range=1.0))
                        patch_ssims.append(structural_similarity(gt_patch, pr_patch, data_range=1.0, channel_axis=2))
                grid_psnr = np.mean(patch_psnrs)
                grid_ssim = np.mean(patch_ssims)
                all_grid_psnrs.append(grid_psnr)
                all_grid_ssims.append(grid_ssim)

                stats_file.write(f"--- {fn} ---\n")
                stats_file.write(f"  PSNR: {img_psnr:.4f}  SSIM: {img_ssim:.4f}  LPIPS: {img_lpips:.4f}\n")
                if img_pu_psnr is not None:
                    stats_file.write(
                        f"  PU-PSNR: {img_pu_psnr:.4f}  PU-SSIM: {img_pu_ssim:.4f}  "
                        f"PU-MS-SSIM: {img_pu_ms_ssim:.4f}  "
                        f"CosineDist: {img_cosdist:.6f}\n"
                    )
                else:
                    stats_file.write(f"  CosineDist: {img_cosdist:.6f}\n")
                stats_file.write(f"  PSNR (3x3 grid): {grid_psnr:.4f}  SSIM (3x3 grid): {grid_ssim:.4f}\n")
                for ci, cn in enumerate(ch_names):
                    gt_ch = batch_np[j, ci]
                    pr_ch = sample_np[j, ci]
                    stats_file.write(f"  GT   {cn}: mean={gt_ch.mean():.4f} min={gt_ch.min():.4f} max={gt_ch.max():.4f}\n")
                    stats_file.write(f"  Pred {cn}: mean={pr_ch.mean():.4f} min={pr_ch.min():.4f} max={pr_ch.max():.4f}\n")
                stats_file.write("\n")
                stats_file.flush()

                if save_timesteps:
                    for t, s in enumerate(samples_disp):
                        tv.utils.save_image(
                            s[j], os.path.join(image_path, f"{fn}_{t}.png")
                        )

                if save_pred:
                    if save_as_hdf5:
                        pred_np_path = os.path.join(
                            inference_output_path, fn + "_pred_u16.hdf5"
                        )
                        create_folder_for_file(pred_np_path)
                        sample_data = sample_np[j].transpose(1, 2, 0)
                        sample_data = (sample_data * 65535).astype(np.uint16)
                        with h5py.File(pred_np_path, "w") as f:
                            f.create_dataset(
                                "rgb",
                                data=sample_data,
                                compression="gzip",
                                compression_opts=9,
                            )
                    else:
                        pred_np_path = os.path.join(
                            inference_output_path, fn + "_pred.npy"
                        )
                        create_folder_for_file(pred_np_path)
                        np.save(pred_np_path, sample_np[j].transpose(1, 2, 0))

                if save_tar:
                    np.save(
                        os.path.join(inference_output_path, fn + "_tar.npy"),
                        batch_np[j].transpose(1, 2, 0),
                    )

                num_samples += 1

            progress.update(task_total_id, advance=1)
            metric_value = metrics_sampling.compute()
            psnr = metric_value["psnr"].item()
            ssim = metric_value["ssim"].item()
            task_total.fields["psnr_rgb"] = f"{psnr:.3f}"
            task_total.fields["ssim_rgb"] = f"{ssim:.4f}"

    batch = {}
    for key, value in metrics_sampling.compute().items():
        print("%s: %.6f" % (key, value.item()))
        batch[key] = value.item()

    with open(metric_path, "w") as f:
        yaml.dump(batch, f)

    stats_file.write("=== Overall Metrics ===\n")
    for key, value in batch.items():
        stats_file.write(f"{key}: {value:.6f}\n")

    if all_grid_psnrs:
        avg_grid_psnr = np.mean(all_grid_psnrs)
        avg_grid_ssim = np.mean(all_grid_ssims)
        stats_file.write(f"\n=== 3x3 Grid Metrics (paper-style) ===\n")
        stats_file.write(f"psnr_3x3grid: {avg_grid_psnr:.6f}\n")
        stats_file.write(f"ssim_3x3grid: {avg_grid_ssim:.6f}\n")
        print(f"psnr_3x3grid: {avg_grid_psnr:.6f}")
        print(f"ssim_3x3grid: {avg_grid_ssim:.6f}")

    with open(config_path, "w") as f:
        yaml.dump(OmegaConf.to_container(cfg), f)

    stats_file.close()

    print("sampling complete")


if __name__ == "__main__":
    main()
