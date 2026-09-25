"""Train ExpandDiff, or retrain it on your own data.

Run from the repository root, after building a split with the scripts in
`preprocessing/`. This reproduces ExpandDiff-P:

    python training/train.py general.is_linear=true general.target_encoding=pu21 \
      general.weight_logl1=0.0 general.weight_pul1=0.0 \
      dataset.train.data_dir=data/scenehdr_pct_512 \
      dataset.val.data_dir=data/scenehdr_pct_512 \
      dataset.train.file_list=SceneHDR_train.txt \
      dataset.train.batch_size=32 general.max_steps=150000 \
      general.lr=2e-4 general.lr_scheduler=cosine general.seed=0 \
      general.suffix=<name> general.check_val_every_n_epoch=10

ExpandDiff-B is the same with `data_dir=data/scenehdr_c95_512`,
`general.max_steps=75000` and `general.lr=1e-4`.
The ablations drop `general.target_encoding=pu21` and/or add
`model.out_tanh=false`. `checkpoints/MANIFEST.md` gives every field of the
released runs.

KEY SETTINGS:

  general.target_encoding   `pu21` trains against the PU21-encoded target;
                            `none` against linear radiance. Sampling and
                            inference must later be told the same thing.
  general.is_linear         must be true for HDR data. It gates the PU metrics
                            and the visualisation transfer function.
  general.suffix            names the experiment directory, which is also built
                            from the data dir, batch size, step budget, loss
                            weights and target encoding. Two runs differing in
                            any of those cannot collide.
  general.lr                is not part of that directory name and defaults to
                            1e-4. Always pass it explicitly: runs that differ
                            only in lr resolve to the same directory.
  weight_logl1, weight_pul1 must be 0.0 with target_encoding=pu21; training
                            refuses to start otherwise.

WHAT IT WRITES, under `experiments/<resolved name>/`:

    checkpoints/last.ckpt       resume point, also what sample.py loads
    training/<step>_*.png       one-step previews, every 500 steps
    validation_sampling/        full DDIM samples on the validation split

Checkpoints are saved every 1000 steps and validation runs every
`general.check_val_every_n_epoch` epochs.
"""
# Make the repository root importable no matter where this is run from.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
from typing import Any

import hydra
import lightning.pytorch as pl
import lightning.pytorch.callbacks as callbacks
import torch
# Override torch.load default to allow checkpoint resumption (PyTorch 2.6+ changed default to weights_only=True)
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, 'weights_only': kwargs.get('weights_only', False)})
import torch.optim.lr_scheduler as lr_scheduler
from aim.pytorch_lightning import AimLogger
from hydra.utils import instantiate
from lightning.pytorch.core import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torchinfo import summary
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from rawdiffusion.datasets.dataset_factory import create_dataset
from rawdiffusion.evaluation.collection import CollectionMetric
from rawdiffusion.evaluation.metrics import (
    LPIPSMetric,
    MSEMetric,
    PearsonMetric,
    PSNRMetric,
    SSIMMetric,
)
from rawdiffusion.resample import create_named_schedule_sampler
from rawdiffusion.gaussian_diffusion_factory import (
    create_gaussian_diffusion,
)
from rawdiffusion.utils import decode_target, get_output_path, linear_to_srgb
from rawdiffusion.config import mod_config


class RAWDiffusionModule(LightningModule):
    def __init__(self, experiment_folder, **hparams) -> None:
        super().__init__()

        self.params = DictConfig(hparams)
        self.log_folder = experiment_folder
        self.save_hyperparameters()

        in_channels = self.params.model.in_channels
        image_size = self.params.general.image_size
        # Width of the guidance the RGB guidance module actually receives: 3
        # normally, 4 when clip_mask_channel appends the mask. The torchinfo
        # summary below builds a dummy input from this.
        guidance_channels = self.params.model.rgb_guidance_module.get("n_colors", 3)

        if self.params.general.get("clip_mask_channel", False):
            nc = self.params.model.rgb_guidance_module.get("n_colors", 3)
            if nc != 4:
                raise ValueError(
                    "clip_mask_channel=true appends a 4th guidance channel, so "
                    "model.rgb_guidance_module.n_colors must be 4 (got "
                    f"{nc}). Pass model/rgb_guidance_module.n_colors=4.")
        self.target_encoding = self.params.general.get("target_encoding", "none")
        if self.target_encoding != "none":
            # log-L1 and PU-L1 both re-curve the target, so refuse either on
            # top of an encoded target.
            bad = [n for n in ("weight_logl1", "weight_pul1")
                   if self.params.general.get(n, 0.0) > 0.0]
            if bad:
                raise ValueError(
                    f"target_encoding={self.target_encoding} double-encodes with "
                    f"{', '.join(bad)}. Pass "
                    + " ".join(f"general.{n}=0.0" for n in bad)
                    + " and keep weight_l1/weight_l2, which are already "
                      "perceptually weighted once the target is PU-encoded.")
            if not self.params.general.is_linear:
                raise ValueError(
                    f"target_encoding={self.target_encoding} requires "
                    "general.is_linear=true (PU is defined on linear "
                    "luminance; encoding sRGB values double-warps the curve).")

        self.model = instantiate(self.params.model, image_size=image_size)
        self.diffusion = create_gaussian_diffusion(**self.params.diffusion)
        self.diffusion_val = create_gaussian_diffusion(**self.params.diffusion_val)
        self.schedule_sampler = create_named_schedule_sampler(
            self.params.general.schedule_sampler, self.diffusion
        )

        summary(
            self.model,
            input_size=[
                (1, in_channels, image_size, image_size),
                (1,),
                (1, guidance_channels, image_size, image_size),
            ],
            depth=2,
        )

    def normalize_inv(self, x):
        return (x + 1) / 2.0

    def save_vis(self, vis, vis_dir, base):
        """Write a debug strip. The strip interleaves encoded panels (target,
        model output) with linear ones (guidance), so it cannot be decoded as a
        whole; for PU targets it is saved as-is."""
        if self.target_encoding != "none":
            to_pil_image(vis.clamp(0, 1)).save(
                os.path.join(vis_dir, base + f"_{self.target_encoding}.png"))
        elif self.params.general.is_linear:
            to_pil_image(vis.clamp(0, 1)).save(os.path.join(vis_dir, base + "_linear.png"))
            to_pil_image(linear_to_srgb(vis)).save(os.path.join(vis_dir, base + "_srgb.png"))
        else:
            to_pil_image(vis.clamp(0, 1)).save(os.path.join(vis_dir, base + ".png"))

    def setup(self, stage: str) -> None:
        self.logger.experiment["hparams"] = self.params

    def forward_step(self, input_data, guidance_input, sampling_seed=None):
        t, weights = self.schedule_sampler.sample(
            input_data.shape[0], self.device, seed=sampling_seed
        )

        losses, extra = self.diffusion.training_losses(
            self.model,
            input_data,
            t,
            model_kwargs=guidance_input,
            weight_l2=self.params.general.weight_l2,
            weight_l1=self.params.general.weight_l1,
            weight_logl1=self.params.general.weight_logl1,
            weight_pul1=self.params.general.get("weight_pul1", 0.0),
            clip_loss_weight=self.params.general.get("clip_loss_weight", 0.0),
            clip_loss_thr=self.params.general.get("clip_loss_thr", 0.05),
        )

        loss = (losses["loss"] * weights).mean()
        metrics = {k: v * weights for k, v in losses.items() if k != "loss"}

        return loss, extra, metrics

    def training_step(self, batch, batch_idx):
        input_data = batch["target_data"]
        guidance_data = batch["guidance_data"]

        if self.params.general.get("clip_mask_channel", False):
            guidance_data = self.append_clip_mask(guidance_data)
        guidance_input = self.preprocess_guidance(guidance_data)

        loss, extra, metrics = self.forward_step(input_data, guidance_input)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        for k, v in metrics.items():
            self.log(
                f"train_{k}",
                v.mean(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        if (
            self.global_step % self.params.general.log_train_images_interval == 0
            and self.global_step >= 0
        ):
            self.log_batch_results(guidance_input, extra)
        if (
            self.global_step % self.params.general.log_train_images_interval == 0
            and self.global_step >= 0
        ):
            self.log_sampling_images(batch)

        return loss

    def on_validation_start(self) -> None:
        self.metrics_sampling = CollectionMetric(
            {
                "mse": MSEMetric(),
                "psnr": PSNRMetric(),
                "ssim": SSIMMetric(),
                "pearson": PearsonMetric(),
                "lpips": LPIPSMetric(input_is_linear=self.params.general.is_linear),
            }
        )

        self.eval_diffusion_process = (
            self.current_epoch + 1
        ) % self.params.general.eval_diffusion_process_interval == 0
        print("validation_start", self.current_epoch, self.eval_diffusion_process)

    def validation_step(self, batch, batch_idx):
        input_data = batch["target_data"]
        guidance_data = batch["guidance_data"]

        if self.params.general.get("clip_mask_channel", False):
            guidance_data = self.append_clip_mask(guidance_data)
        guidance_input = self.preprocess_guidance(guidance_data)

        sampling_seed = 123 + batch_idx
        loss, extra, metrics = self.forward_step(
            input_data, guidance_input, sampling_seed=sampling_seed
        )

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        val_steps_per_epoch = self.trainer.num_val_batches[0]
        sampling_interval = max(
            1, val_steps_per_epoch // self.params.general.val_sampling_frequency
        )

        sampling = batch_idx % sampling_interval == 0
        sampling_log_images = (
            batch_idx // sampling_interval
        ) % self.params.general.log_val_sampling_images_interval == 0

        if sampling and sampling_log_images:
            filename = f"{batch_idx:04d}_{(self.global_step):06d}.png"
            self.log_batch_results(
                guidance_input, extra, mode="validation", filename=filename
            )

        if self.eval_diffusion_process and sampling:
            filename = f"{batch_idx:04d}_{(self.global_step):06d}.png"
            save_results = sampling_log_images
            raw_generated = self.log_sampling_images(
                batch,
                mode_name="validation_sampling",
                filename=filename,
                sampling_seed=sampling_seed,
                save_results=save_results,
            )
            input_data_device = input_data.to(raw_generated.device)

            # decode first: val metrics then mean the same thing whether or
            # not the target is encoded
            self.metrics_sampling.update(
                decode_target(self.normalize_inv(input_data_device),
                              self.target_encoding),
                decode_target(self.normalize_inv(raw_generated),
                              self.target_encoding),
            )

    def on_validation_epoch_end(self) -> None:
        if self.eval_diffusion_process:
            for k, v in self.metrics_sampling.compute().items():
                self.log(f"val_sampling_{k}", v)

    def append_clip_mask(self, guidance_data):
        """Concatenate the soft clip mask onto the guidance.

        The mask tells the model where the input carries no information. It is
        computed from the guidance only.
        """
        from rawdiffusion.gaussian_diffusion import clip_soft_mask

        m = clip_soft_mask(guidance_data,
                           self.params.general.get("clip_loss_thr", 0.05))
        return torch.cat([guidance_data, m * 2.0 - 1.0], dim=1)

    def preprocess_guidance(self, guidance_data):
        guidance_input = {}

        drop_rate = self.params.general.drop_rate
        bs = guidance_data.shape[0]
        if self.training and drop_rate > 0.0:
            mask = (
                (torch.rand([bs, 1, 1, 1]) > drop_rate).float().to(guidance_data.device)
            )
            guidance_data = guidance_data * mask

        guidance_input["guidance_data"] = guidance_data

        return guidance_input

    def log_batch_results(
        self, model_kwargs, return_dict, mode="training", filename=None
    ):
        x_start = return_dict["x_start"]
        x_t = return_dict["x_t"]
        model_output = return_dict["model_output"]
        target = return_dict["target"]
        guidance_data = model_kwargs["guidance_data"]

        vis = torch.concatenate(
            [
                x_start,
                # RGB only: with clip_mask_channel the guidance carries a 4th
                # (mask) channel, and this strip concatenates along WIDTH, which
                # still requires every panel to agree on channels. A no-op slice
                # in the default 3-channel case.
                guidance_data.to(x_start.device)[:, :3],
                x_t,
                model_output,
                target,
                model_output - target,
            ],
            dim=3,
        )
        vis = torch.clamp(vis, -1, 1)
        vis = make_grid(vis, nrow=1)
        vis = self.normalize_inv(vis)

        if filename is None:
            filename = f"{(self.global_step):06d}.png"

        base = os.path.splitext(filename)[0]
        vis_dir = os.path.join(self.log_folder, mode)
        os.makedirs(vis_dir, exist_ok=True)
        self.save_vis(vis, vis_dir, base)

    def log_sampling_images(
        self,
        batch,
        mode_name="training_sampling",
        filename=None,
        sampling_seed=None,
        save_results=True,
    ):
        input_data = batch["target_data"]
        guidance_data = batch["guidance_data"]

        if self.params.general.get("clip_mask_channel", False):
            guidance_data = self.append_clip_mask(guidance_data)
        guidance_input = self.preprocess_guidance(guidance_data)

        bs, _, h, w = input_data.shape

        use_ddim = True
        clip_denoised = True
        diffusion = self.diffusion_val

        g = torch.Generator(device=self.device)
        if sampling_seed is not None:
            g.manual_seed(sampling_seed)

        with torch.inference_mode():
            shape = (bs, self.params.model.in_channels, h, w)
            noise = torch.randn(*shape, device=self.device, generator=g)

            sample_fn = (
                diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                self.model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                model_kwargs=guidance_input,
                progress=True,
            )

        d = sample.device

        result_out = sample

        if save_results:
            # guidance RGB only -- see the note on the training-viz strip above
            vis = torch.concat(
                [input_data.to(d), guidance_data.to(d)[:, :3], sample], dim=3)
            vis = torch.clamp(vis, -1, 1)
            vis = self.normalize_inv(vis)

            vis = make_grid(vis, nrow=1)

            if filename is None:
                filename = f"{(self.global_step):06d}.png"

            base = os.path.splitext(filename)[0]
            vis_dir = os.path.join(self.log_folder, mode_name)
            os.makedirs(vis_dir, exist_ok=True)
            self.save_vis(vis, vis_dir, base)

        return result_out

    def configure_optimizers(self) -> Any:
        optimizer = AdamW(
            self.parameters(),
            lr=self.params.general.lr,
            weight_decay=self.params.general.weight_decay,
        )

        if self.params.general.lr_scheduler == "linear":
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.params.general.max_steps,
            )
        elif self.params.general.lr_scheduler == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.params.general.max_steps, eta_min=0.0
            )
        else:
            raise ValueError(
                f"Unknown lr_scheduler: {self.params.general.lr_scheduler}"
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


@hydra.main(version_base="1.3", config_path="configs", config_name="rawdiffusion")
def main(cfg: DictConfig) -> None:
    mod_config(cfg)
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.general.seed)

    aim_logger = AimLogger(
        experiment="diffusion",
        train_metric_prefix=None,
        test_metric_prefix=None,
        val_metric_prefix=None,
    )

    experiment_folder = get_output_path(cfg)
    print(f"experiment_folder: {experiment_folder}")

    print("creating data loader...")
    target_encoding = cfg.general.get("target_encoding", "none")
    data_train = create_dataset(
        **cfg.dataset.train, seed=cfg.general.seed,
        patch_size=cfg.general.image_size, target_encoding=target_encoding
    )
    data_val = create_dataset(
        **cfg.dataset.val,
        seed=cfg.general.seed,
        patch_size=cfg.general.image_size,
        permutate_once=True,
        target_encoding=target_encoding,
    )

    raw_module = RAWDiffusionModule(experiment_folder=experiment_folder, **cfg)

    trainer_callbacks = [
        callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    if cfg.general.checkpoint:
        checkpoint_path = os.path.join(
            experiment_folder,
            "checkpoints",
        )
        checkpoint_cb = callbacks.ModelCheckpoint(
            dirpath=checkpoint_path,
            save_last=True,
            every_n_train_steps=cfg.general.save_interval,
            enable_version_counter=False,
        )
        trainer_callbacks.append(checkpoint_cb)
        print("checkpoint", experiment_folder)

    trainer = pl.Trainer(
        # "auto" uses the GPU when one is present, and lets a short smoke test
        # run on a CPU-only machine.
        accelerator="auto",
        devices=1,
        max_steps=cfg.general.max_steps,
        logger=aim_logger,
        callbacks=trainer_callbacks,
        enable_checkpointing=cfg.general.checkpoint,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epoch,
        limit_train_batches=1000,
    )

    # Resume from last checkpoint if it exists
    ckpt_resume_path = None
    if cfg.general.checkpoint:
        last_ckpt = os.path.join(experiment_folder, "checkpoints", "last.ckpt")
        if os.path.exists(last_ckpt):
            ckpt_resume_path = last_ckpt
            print(f"Resuming from checkpoint: {last_ckpt}")
        else:
            print("No checkpoint found, starting from scratch")

    trainer.fit(raw_module, data_train, data_val, ckpt_path=ckpt_resume_path)


if __name__ == "__main__":
    main()
