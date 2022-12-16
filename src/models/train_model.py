import math
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Tuple

import gin
import numpy as np
import tensorflow as tf  # noqa: F401

# needed to make summarywriter load without error
import torch
from loguru import logger
from numpy import Inf
from ray import tune
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from src.data import data_tools
from src.models.metrics import Metric
from src.typehinting import GenericModel


@gin.configurable
def trainloop(
    epochs: int,
    model: GenericModel,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    loss_fn: Callable,
    metrics: List[Metric],
    train_dataloader: Iterator,
    test_dataloader: Iterator,
    log_dir: Path,
    train_steps: int,
    eval_steps: int,
    patience: int = 10,
    factor: float = 0.9,
    tunewriter: bool = False,
) -> GenericModel:
    """

    Args:
        epochs (int) : Amount of runs through the dataset
        model: A generic model with a .train() and .eval() method
        optimizer : an uninitialized optimizer class. Eg optimizer=torch.optim.Adam
        learning_rate (float) : floating point start value for the optimizer
        loss_fn : A loss function
        metrics (List[Metric]) : A list of callable metrics.
            Assumed to have a __repr__ method implemented, see src.models.metrics
            for Metric details
        train_dataloader, test_dataloader (Iterator): data iterators
        log_dir (Path) : where to log stuff when not using the tunewriter
        train_steps, eval_steps (int) : amount of times the Iterators are called for a
            new batch of data.
        patience (int): used for the ReduceLROnPlatues scheduler. How many epochs to
            wait before dropping the learning rate.
        factor (float) : fraction to drop the learning rate with, after patience epochs
            without improvement in the loss.
        tunewriter (bool) : when running experiments manually, this should
            be False (default). If false, a subdir is created
            with a timestamp, and a SummaryWriter is invoked to write in
            that subdir for Tensorboard use.
            If True, the logging is left to the ray.tune.report


    Returns:
        _type_: _description_
    """

    optimizer_: torch.optim.Optimizer = optimizer(
        model.parameters(), lr=learning_rate
    )  # type: ignore

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_,
        factor=factor,
        patience=patience,
    )

    if not tunewriter:
        log_dir = data_tools.dir_add_timestamp(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        write_gin(log_dir, gin.config_str())

        images, _ = next(iter(train_dataloader))
        if len(images.shape) == 4:
            grid = make_grid(images)
            writer.add_image("images", grid)
        writer.add_graph(model, images)

    for epoch in tqdm(range(epochs)):
        train_loss = trainbatches(
            model, train_dataloader, loss_fn, optimizer_, train_steps
        )

        metric_dict, test_loss = evalbatches(
            model, test_dataloader, loss_fn, metrics, eval_steps
        )

        scheduler.step(test_loss)

        if tunewriter:
            tune.report(
                iterations=epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                **metric_dict,
            )
        else:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            for m in metric_dict:
                writer.add_scalar(f"metric/{m}", metric_dict[m], epoch)
            lr = [group["lr"] for group in optimizer_.param_groups][0]
            writer.add_scalar("learning_rate", lr, epoch)
            metric_scores = [f"{v:.4f}" for v in metric_dict.values()]
            logger.info(
                f"Epoch {epoch} train {train_loss:.4f} test {test_loss:.4f} metric {metric_scores}"  # noqa E501
            )

    return model