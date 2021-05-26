import torch
import pytorch_lightning as pl
from pytorchvideo.models import create_res_basic_head

from model import Classifier
from data import make_ucf11_datamodule


def make_slowr50_finetuner(num_labels, freeze_backbone=True):
    '''Init Pretrained Model, freeze its backbone, and replace its classification head'''
    model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
    model.blocks[:-1].requires_grad_(not freeze_backbone)
    model.blocks[-1] = create_res_basic_head(in_features=2048, out_features=num_labels)
    return model


def main():
    pl.seed_everything(42)
    dm = make_ucf11_datamodule(batch_size=8, num_workers=2)
    model = make_slowr50_finetuner(dm.num_labels)
    classifier = Classifier(model, lr=2e-4)
    trainer = pl.Trainer(gpus=1, precision=16, max_epochs=4)
    trainer.fit(classifier, dm)


if __name__ == '__main__':
    main()
