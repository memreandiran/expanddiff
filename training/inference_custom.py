"""Run a checkpoint over a folder of image files, with no ground truth required.

It takes a checkpoint path directly, so it works with a released checkpoint.
`sample.py` instead needs a preprocessed split.

Pass both flags below with every released ExpandDiff checkpoint:

  python training/inference_custom.py \
    --checkpoint checkpoints/expanddiff_p_150k.ckpt \
    --input_dir  <folder of png/jpg> \
    --output_dir <folder for results> \
    --linear_model --target_encoding pu21 --save_guidance

  --linear_model      the checkpoint works in linear radiance: 8-bit inputs are
                      linearised with the inverse sRGB curve before being fed,
                      and predictions come back linear. Required for every
                      released checkpoint.
  --target_encoding   must match how the checkpoint was trained: `pu21` for
                      ExpandDiff-P, ExpandDiff-B and the PU21 ablation, `none`
                      for ExpandDiff-D and the two linear ablations. Getting it
                      wrong produces a plausible but wrongly-toned image, so the
                      script prints a warning when it disagrees with the
                      checkpoint. `checkpoints/MANIFEST.md` lists it per file.
  --input_is_linear   the input files are already linear; skip the conversion.
  --save_guidance     also write what the model was actually given, which is
                      what to look at when a result is unexpected.

There is no flag for the bounded head: whether the model ends in tanh is read
from the checkpoint, so the unbounded ablations rebuild correctly as they are.

WHAT YOU GET, per input image:

  <name>_generated_linear.tiff   the reconstruction: float32 linear radiance,
                                 1.0 = 1000 cd/m^2, so multiply by 1000 for
                                 cd/m^2. This is the file to measure or to take
                                 into an HDR pipeline.
  <name>_generated_srgb.png      an 8-bit preview of the same data, for looking
                                 at on an ordinary display.
  <name>_guidance_{linear.tiff,srgb.png}   with --save_guidance, the input as
                                 the model received it.

Images of any size work: the input is reflect-padded to a multiple of 64 for the
U-Net and cropped back before saving. Memory is the only limit.

The model reconstructs what clipping removed, so it expects a clipped input. An
image with no blown or crushed pixels is outside the training distribution and
has little for the model to do.
"""

# Make the repository root importable no matter where this is run from.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
from pathlib import Path

import cv2
import imageio
import numpy as np
import tifffile
import torch
from hydra import compose, initialize

from rawdiffusion.config import mod_config
from rawdiffusion.gaussian_diffusion_factory import create_gaussian_diffusion
from rawdiffusion.utils import (TARGET_ENCODINGS, decode_target,
                                linear_to_srgb, srgb_to_linear)
from train import RAWDiffusionModule


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def save_float_tiff(tensor_chw, path):
    """Save a (C, H, W) torch tensor in [0, 1] as a float32 TIFF
    with no quantization. Used for linear-RGB guidance and generated
    images so downstream HDR consumers get unrounded values."""
    arr = tensor_chw.detach().cpu().to(torch.float32).numpy()
    arr = np.transpose(arr, (1, 2, 0))  # (H, W, C)
    tifffile.imwrite(str(path), arr, photometric="rgb")


def pad_to_multiple(tensor, multiple=64):
    """Pad (B, C, H, W) tensor so H and W are multiples of `multiple`."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, (h, w)


def load_rgb(path):
    """Load an image as uint8 sRGB (H, W, 3)."""
    img = imageio.imread(str(path))
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config_name", default="rawdiffusion_sample")
    parser.add_argument("--timesteps", default="ddim24")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Integer divisor applied to the input before inference. "
                             "Set to 2 if your inputs are full-res HDR+ JPGs to match "
                             "the training pipeline.")
    parser.add_argument("--resize_to", type=int, default=None,
                        help="If set, scale long edge to this value (no upscaling).")
    parser.add_argument("--save_guidance", action="store_true",
                        help="Also save the guidance image actually fed to the model.")
    parser.add_argument("--linear_model", action="store_true",
                        help="The checkpoint was trained on linear-RGB data (the "
                             "current_pipeline_linear_RGB.py pipeline). When set, "
                             "sRGB inputs are converted to linear before being fed, "
                             "and the prediction is saved both as raw linear and as "
                             "sRGB-gamma-encoded for viewing. Off by default (8-bit "
                             "sRGB pipeline).")
    parser.add_argument("--target_encoding", default="none",
                        choices=list(TARGET_ENCODINGS),
                        help="Must match general.target_encoding of the run the "
                             "checkpoint came from. 'pu21' decodes the "
                             "prediction back to linear radiance; getting this "
                             "wrong silently produces a wrongly-toned image.")
    parser.add_argument("--input_is_linear", action="store_true",
                        help="Only meaningful with --linear_model: if set, the input "
                             "PNG is already linear (skip the sRGB->linear conversion).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with initialize(config_path="configs", version_base="1.3"):
        cfg = compose(config_name=args.config_name)
    mod_config(cfg)
    cfg.diffusion_val.timestep_respacing = args.timesteps

    # Architecture settings come from the checkpoint itself, so a checkpoint
    # trained without the bounded head is rebuilt without it.
    ckpt_cfg = torch.load(args.checkpoint, map_location="cpu",
                          weights_only=False).get("hyper_parameters", {})
    ckpt_model = ckpt_cfg.get("model", {}) if isinstance(ckpt_cfg, dict) else {}
    if "out_tanh" in ckpt_model and ckpt_model["out_tanh"] != cfg.model.out_tanh:
        print(f"checkpoint was trained with out_tanh={ckpt_model['out_tanh']}; "
              f"using that instead of the config default {cfg.model.out_tanh}")
        cfg.model.out_tanh = ckpt_model["out_tanh"]
    ckpt_enc = (ckpt_cfg.get("general", {}) or {}).get("target_encoding", "none") \
        if isinstance(ckpt_cfg, dict) else "none"
    if ckpt_enc != args.target_encoding:
        print(f"WARNING: checkpoint was trained with target_encoding={ckpt_enc}, "
              f"but --target_encoding {args.target_encoding} was passed. The "
              f"prediction will be decoded with the wrong curve.")

    # `checkpoint_path` is a config key of sample.py's, and its name collides
    # with load_from_checkpoint's own first argument, so it never goes through.
    mod_kwargs = {k: v for k, v in cfg.items() if k != "checkpoint_path"}
    raw_module = RAWDiffusionModule.load_from_checkpoint(
        args.checkpoint, experiment_folder=str(output_dir), **mod_kwargs
    )
    raw_module.eval().cuda()
    diffusion = create_gaussian_diffusion(**cfg.diffusion_val)
    in_channels = cfg.model.in_channels

    input_dir = Path(args.input_dir)
    input_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not input_paths:
        print(f"No images found in {input_dir} with extensions {sorted(IMAGE_EXTENSIONS)}")
        return

    for img_path in input_paths:
        print(f"Processing: {img_path.name}")

        try:
            rgb = load_rgb(img_path)
        except Exception as e:
            print(f"  Skipping {img_path.name} ({e})")
            continue

        if args.downsample > 1:
            h, w = rgb.shape[:2]
            rgb = cv2.resize(
                rgb,
                (w // args.downsample, h // args.downsample),
                interpolation=cv2.INTER_AREA,
            )

        if args.resize_to:
            h, w = rgb.shape[:2]
            scale = args.resize_to / max(h, w)
            if scale < 1.0:
                rgb = cv2.resize(
                    rgb,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

        rgb_tensor = (
            torch.from_numpy(rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .cuda()
        )  # [0, 1] sRGB-encoded (PNG values, native form)

        # Convert input to whatever space the model was trained on.
        if args.linear_model and not args.input_is_linear:
            rgb_tensor = srgb_to_linear(rgb_tensor)

        if args.save_guidance:
            guide = rgb_tensor[0].clamp(0, 1).cpu()
            if args.linear_model:
                # Linear guidance is saved as a float32 TIFF to avoid
                # quantization to uint8.
                save_float_tiff(
                    guide,
                    output_dir / f"{img_path.stem}_guidance_linear.tiff",
                )
                imageio.imwrite(
                    output_dir / f"{img_path.stem}_guidance_srgb.png",
                    (linear_to_srgb(guide).numpy().transpose(1, 2, 0) * 255).astype(np.uint8),
                )
            else:
                imageio.imwrite(
                    output_dir / f"{img_path.stem}_guidance.png",
                    (guide.numpy().transpose(1, 2, 0) * 255).astype(np.uint8),
                )

        rgb_tensor = rgb_tensor * 2 - 1
        rgb_tensor, (h_orig, w_orig) = pad_to_multiple(rgb_tensor, 64)

        guidance = {
            k: v.cuda()
            for k, v in raw_module.preprocess_guidance(rgb_tensor).items()
        }
        noise = torch.randn(
            1, in_channels, rgb_tensor.shape[2], rgb_tensor.shape[3], device="cuda"
        )

        with torch.inference_mode():
            for step in diffusion.ddim_sample_loop_progressive(
                raw_module.model,
                noise.shape,
                noise=noise,
                model_kwargs=guidance,
                clip_denoised=True,
            ):
                sample = step["sample"]

        sample = (sample[:, :, :h_orig, :w_orig] + 1) / 2.0
        # decode before clamping: the clamp is a display guard, and clamping an
        # encoded value first would flatten highlights the decode should keep
        sample = decode_target(sample, args.target_encoding)
        sample = sample[0].clamp(0, 1).cpu()

        if args.linear_model:
            # Linear prediction goes out as float32 TIFF (no quantization);
            # sRGB-encoded view stays a PNG for normal viewing.
            save_float_tiff(
                sample, output_dir / f"{img_path.stem}_generated_linear.tiff"
            )
            sample_srgb = (
                linear_to_srgb(sample).numpy().transpose(1, 2, 0) * 255
            ).astype(np.uint8)
            imageio.imwrite(output_dir / f"{img_path.stem}_generated_srgb.png", sample_srgb)
        else:
            sample_srgb = (sample.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            imageio.imwrite(output_dir / f"{img_path.stem}_generated.png", sample_srgb)


if __name__ == "__main__":
    main()
