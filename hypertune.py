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

def train(config: Dict, checkpoint_dir=None):

    data_dir = "../../data/raw"
    with FileLock(data_dir / ".lock"):
        trainloader, testloader = make_dataset.get_flowers(
            data_dir=data_dir, split=0.8, batchsize=32
        )

    accuracy = metrics.Accuracy()
    model = rnn_models.GRUmodel(config) #Model moet nog aangepast worden naar Resnet.

    model = train_model.trainloop( #Trainloop moet nog toegevoegd worden.
        epochs=50,
        model=model,
        optimizer=torch.optim.Adam,
        learning_rate=1e-3,
        loss_fn=torch.nn.CrossEntropyLoss(),
        metrics=[accuracy],
        train_dataloader=trainloader,
        test_dataloader=testloader,
        log_dir=".",
        train_steps=len(trainloader),
        eval_steps=len(testloader),
        patience=5,
        factor=0.5,
        tunewriter=True,
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
