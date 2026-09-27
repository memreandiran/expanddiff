import os

from torch.utils.data import DataLoader
from rawdiffusion.datasets.transforms import ImageTransforms
from .raw_image_dataset import RGBImageDataset


def create_dataset(
    *,
    data_dir,
    file_list,
    batch_size,
    seed,
    is_train=True,
    transform=True,
    permutate_once=False,
    resample_dataset_size=None,
    patch_size=256,
    max_items=None,
    num_workers=8,
    target_encoding="none",
    **_ignored,
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    if transform:
        transforms = ImageTransforms(
            patch_size=patch_size,
            is_train=is_train,
        )
    else:
        transforms = None

    if max_items is not None:
        name, ext = os.path.splitext(file_list)
        file_list = f"{name}_{max_items}_{seed}{ext}"

    dataset = RGBImageDataset(
        file_list=file_list,
        dataset_path=data_dir,
        transforms=transforms,
        target_encoding=target_encoding,
    )

    if permutate_once:
        from .dataset_wrapper import PermutedDataset

        dataset = PermutedDataset(dataset, seed=123)

    if resample_dataset_size is not None:
        from .dataset_wrapper import RandomSampleDataset

        dataset = RandomSampleDataset(dataset, n=resample_dataset_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        drop_last=is_train,
    )
    return loader
