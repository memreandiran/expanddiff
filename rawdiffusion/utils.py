import os
import torch


def gamma_correction(t, gamma=1.0 / 5):
    t = t.clip(0, 1)
    t = t**gamma
    return t


def linear_to_srgb(t):
    """Linear RGB tensor in [0, 1] -> sRGB-gamma-encoded tensor in [0, 1].

    Uses the proper sRGB EOTF-inverse (piecewise: linear toe + 2.4 power).
    Use this on linear-RGB targets/predictions before saving as PNG so that
    standard image viewers display them with correct tone.
    """
    t = t.clamp(0.0, 1.0)
    return torch.where(t <= 0.0031308, t * 12.92, 1.055 * t ** (1.0 / 2.4) - 0.055)


def srgb_to_linear(t):
    """sRGB-gamma-encoded tensor in [0, 1] -> linear RGB tensor in [0, 1].

    Inverse of `linear_to_srgb`. Use on sRGB inputs (e.g. user-supplied PNGs)
    before feeding them to a model trained on linear data.
    """
    t = t.clamp(0.0, 1.0)
    return torch.where(t <= 0.04045, t / 12.92, ((t + 0.055) / 1.055) ** 2.4)


# --------------------------------------------------------------------------- #
# Target encodings (opt-in; "none" is the default and leaves everything as-is)
#
# The tanh head can only emit a bounded value, which is fine for
# display-referred data in [0, 1] but cannot represent scene-referred radiance.
# Training against pu21(L) instead of L removes that limit without touching the
# architecture: the target is bounded by construction, and decoding at
# inference recovers ~17.6 stops (0.005 to 1000 cd/m^2). LEDiff does the same
# thing with a log-space decoder.
#
# These mirror pu21_encode_np / pu21_decode_np in datasets/bracket_ops.py --
# keep the two in step, they are the encode and decode ends of one pipeline.
# --------------------------------------------------------------------------- #
_PU21_P = (0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627,
           0.09150303166, 0.9099517204, 596.3148142)


def _pu21_vmax(l_peak):
    p1, p2, p3, p4, p5, p6, p7 = _PU21_P
    yq = float(l_peak) ** p4
    return p7 * (((p1 + p2 * yq) / (1 + p3 * yq)) ** p5 - p6)


def pu21_encode(t, l_peak=1000.0):
    """Linear tensor in [0, 1] -> PU21 tensor in [0, 1]."""
    p1, p2, p3, p4, p5, p6, p7 = _PU21_P
    y = (t.double() * l_peak).clamp(min=0.005)
    yp = y**p4
    v = (p7 * (((p1 + p2 * yp) / (1 + p3 * yp)) ** p5 - p6)).clamp(min=0.0)
    return (v / _pu21_vmax(l_peak)).clamp(0.0, 1.0).to(t.dtype)


def pu21_decode(t, l_peak=1000.0):
    """Inverse of `pu21_encode`: PU21 prediction -> linear radiance in [0, 1].

    This is where the bounded model output turns back into high dynamic range,
    so it is the one step that must not be skipped when sampling a model
    trained with `general.target_encoding=pu21`.
    """
    p1, p2, p3, p4, p5, p6, p7 = _PU21_P
    v = t.double().clamp(0.0, 1.0) * _pu21_vmax(l_peak)
    r = (v / p7 + p6) ** (1.0 / p5)          # = (p1 + p2*y^p4) / (1 + p3*y^p4)
    yp = ((p1 - r) / (p3 * r - p2)).clamp(min=0.0)
    y = yp ** (1.0 / p4)
    return (y / l_peak).clamp(0.0, 1.0).to(t.dtype)


# The OFFICIAL PU21 *metric* convention, from gfxdisp/pu21 (matlab/
# pu21_encoder.m + pu21_metric.m). Deliberately different from pu21_encode
# above, and both are needed:
#
#   pu21_encode        normalises to [0,1] with l_peak. This is the TARGET
#                      encoding -- a bounded range is the entire point -- and
#                      it is the basis every LEDiff-comparison number in
#                      LEDIFF_COMPARISON.md was computed on. Do not change it.
#   pu21_encode_metric absolute cd/m^2 clamped to [0.005, 10000], RAW encode
#                      (no /vmax), so 100 nit maps to ~256 and PSNR is taken
#                      against peak 256. This is the scale AIM 2025 and the
#                      PU21 papers report.
PU21_L_MIN, PU21_L_MAX, PU21_PEAK = 0.005, 10000.0, 256.0


def pu21_encode_metric(t, peak_nits=1000.0):
    """Relative [0,1] -> raw PU21 units, matching the reference implementation.

    `peak_nits` says what 1.0 means in cd/m^2; our median-anchored targets use
    1000, which sits inside the encoder's valid [0.005, 10000] range.
    """
    p1, p2, p3, p4, p5, p6, p7 = _PU21_P
    y = (t.double() * peak_nits).clamp(PU21_L_MIN, PU21_L_MAX)
    yp = y**p4
    return (p7 * (((p1 + p2 * yp) / (1 + p3 * yp)) ** p5 - p6)).clamp(min=0.0)


TARGET_ENCODINGS = ("none", "pu21")


def decode_target(t, encoding):
    """Undo whatever `general.target_encoding` applied to the target.

    Identity for "none", so calling it unconditionally on the existing
    pipelines changes nothing.
    """
    if encoding in (None, "none"):
        return t
    if encoding == "pu21":
        return pu21_decode(t)
    raise ValueError(f"unknown target_encoding: {encoding}")


def create_folder_for_file(file_path):
    folder = os.path.dirname(file_path)
    os.makedirs(folder, exist_ok=True)


def parts_to_str(parts, delimiter="_"):
    return delimiter.join([str(p) for p in parts if p is not None])


def get_rgb_guidance_module_key(args):
    if args is None:
        return None

    model_name = args._target_.split(".")[-1]

    parts = [
        model_name,
    ]

    if model_name == "EDSR":
        parts += [args.n_resblocks, args.n_feats, args.bn]
    elif model_name == "RRDBNet":
        parts += [args.nf, args.nb]
    elif model_name == "NAFNet":
        parts += [
            args.width,
            "E" + parts_to_str(args.enc_blk_nums, "-"),
            args.middle_blk_num,
            "D" + parts_to_str(args.dec_blk_nums, "-"),
        ]
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return parts_to_str(parts)


def get_output_path(args):
    model_name = args.model._target_.split(".")[-1]
    model_params = args.model
    rgb_guidance_module_args = args.model.rgb_guidance_module
    rgb_guidance_module_key = get_rgb_guidance_module_key(rgb_guidance_module_args)
    train_split = os.path.splitext(os.path.basename(args.dataset.train.file_list))[
        0
    ].replace("_train", "")

    if args.dataset.train.max_items is not None:
        mi = args.dataset.train.max_items
        name, _ = os.path.splitext(args.dataset.train.file_list)
        train_split = f"{name}_{mi}_{args.general.seed}"

    parts = [
        os.path.basename(os.path.normpath(args.dataset.train.data_dir)),
        train_split,
        f"R{args.dataset.train.resample_dataset_size}"
        if args.dataset.train.resample_dataset_size is not None
        else None,
        args.general.image_size,
        args.dataset.train.batch_size,
        f"{args.general.max_steps // 1000}k",
        "rawdiffusion",
        args.diffusion.steps,
        args.diffusion.noise_schedule,
        "sigma" if args.diffusion.learn_sigma else None,
        "predict_noise" if not args.diffusion.predict_xstart else None,
        "model",
        model_name,
        args.model.model_channels,
        args.model.num_head_channels,
        args.model.num_res_blocks,
        args.model.norm_num_groups,
        "A" + parts_to_str(args.model.attention_resolutions, "-")
        if args.model.attention_resolutions
        else "noatt",
        "d{:.1f}".format(args.general.drop_rate)
        if args.general.drop_rate > 0.0
        else None,
        "ld{:.1f}".format(args.model.latent_drop_rate)
        if args.model.latent_drop_rate > 0.0
        else None,
        rgb_guidance_module_key,
        model_params.conditional_block_name
        if model_params.conditional_block_name != "RGBGuidedResidualBlock"
        else None,
        "midatt" if args.model.mid_attention else None,
        f"l2{args.general.weight_l2}" if args.general.weight_l2 > 0.0 else None,
        f"l1{args.general.weight_l1}" if args.general.weight_l1 > 0.0 else None,
        f"logl1{args.general.weight_logl1}"
        if args.general.weight_logl1 > 0.0
        else None,
        f"wd{args.general.weight_decay}" if args.general.weight_decay > 0.0 else None,
        f"clipw{args.general.get('clip_loss_weight', 0.0)}"
        if args.general.get("clip_loss_weight", 0.0) > 0.0 else None,
        "clipch" if args.general.get("clip_mask_channel", False) else None,
        # only appended when non-default, so existing runs keep their folders
        f"enc{args.general.get('target_encoding', 'none')}"
        if args.general.get("target_encoding", "none") != "none"
        else None,
        args.general.lr_scheduler,
        "bl" if args.general.min_mode == "black_level" else "mv",
        "tanh" if args.model.out_tanh else None,
        args.general.suffix,
        args.general.seed,
    ]

    experiment_name = parts_to_str(parts)
    return os.path.join("experiments", experiment_name)
