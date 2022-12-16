from src.data import make_dataset
from src.models import rnn_models, metrics, train_model
from src.settings import SearchSpace
from pathlib import Path
from ray.tune import JupyterNotebookReporter
from ray import tune
import torch
import ray
from typing import Dict
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from loguru import logger
from filelock import FileLock
import torchvision
import torch.nn as nn

def train(config: Dict, checkpoint_dir=None):

    data_dir = "../../data/raw"
    with FileLock(data_dir / ".lock"):
        trainloader, testloader = make_dataset.get_flowers(
            data_dir=data_dir, split=0.8, batchsize=32
        )

    accuracy = metrics.Accuracy()
    resnet = torchvision.models.resnet18(pretrained=True)

    for name, param in resnet.named_parameters():
        param.requires_grad = False
    
    in_features = resnet.fc.in_features
    
    resnet.fc = nn.Sequential(
    nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 5)
)

    resnet = train_model.trainloop(
        epochs=10,
        model=resnet,
        metrics=[accuracy],
        optimizer=torch.optim.Adam,
        learning_rate=0.01,
        loss_fn=nn.CrossEntropyLoss(),
        train_dataloader=train_datastream,
        test_dataloader=test_datastream,
        log_dir="../../models/resnet",
        eval_steps=15,
        train_steps=25,
        patience=2,
        factor=0.5,
    )


if __name__ == "__main__":
    ray.init()

    config = SearchSpace( #Searchspace moet nog geïmporteerd worden.
        input_size=3,
        output_size=20,
        tune_dir=Path("models/ray").resolve(),
        data_dir=Path("data/external/gestures-dataset").resolve(),
    )

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=50,
        reduction_factor=4,
        stop_last_trials=False,
    )
    bohb_search = TuneBOHB()

    analysis = tune.run(
        train,
        config=config.dict(),
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        local_dir=config.tune_dir,
        num_samples=50,
        search_alg=bohb_search,
        scheduler=bohb_hyperband,
        verbose=1,
    )

    ray.shutdown()
