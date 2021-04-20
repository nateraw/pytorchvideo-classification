import pytorch_lightning as pl

from data import UCF11
from model import Classifier


def main(
    batch_size: int = 32,
    num_workers: int = 8,
    num_holdout_scenes: int = 2,
    learning_rate: float = 2e-4, 
    seed: int = 42,
    gpus: int = 1,
    precision: int = 16,
    max_epochs: int = 5
):
    pl.seed_everything(seed)
    data = UCF11(
        batch_size=batch_size,
        num_workers=num_workers,
        num_holdout_scenes=num_holdout_scenes
    )
    classifier = Classifier(num_classes=data.num_classes, lr=learning_rate)
    trainer = pl.Trainer(gpus=gpus, precision=precision, max_epochs=max_epochs)
    trainer.fit(classifier, data)
    return data, classifier, trainer


if __name__ == '__main__':
    data, classifier, trainer= main()
    # import typer
    # typer.run(main)
