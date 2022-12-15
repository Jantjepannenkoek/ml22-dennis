from __future__ import annotations

from pathlib import Path

import tensorflow as tf
import torch
from loguru import logger

Tensor = torch.Tensor


def get_flowers(data_dir: Path) -> Path:
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"  # noqa: E501
    image_folder = Path(data_dir) / "datasets/flower_photos"
    if not image_folder.exists():
        image_folder = tf.keras.utils.get_file(
            "flower_photos", origin=dataset_url, untar=True, cache_dir=data_dir
        )
        image_folder = Path(image_folder)
        logger.info(f"Data is downloaded to {image_folder}.")
    else:
        logger.info(f"Dataset already exists at {image_folder}")
    return image_folder
