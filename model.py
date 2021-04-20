import pytorch_lightning as pl
import torch
from pytorchvideo.models.head import create_res_basic_head
from torch import nn
from torch.optim import Adam


class Classifier(pl.LightningModule):
    def __init__(self, num_classes=11, lr=2e-4):
        super().__init__()
        self.save_hyperparameters()

        # Backbone
        resnet = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[0][:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Head
        self.head = create_res_basic_head(in_features=2048, out_features=self.hparams.num_classes)

        # Metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        if isinstance(x, dict):
            x = x["video"]
        feats = self.backbone(x)
        return self.head(feats)

    def shared_step(self, batch, split="train"):
        x = batch["video"]
        y = batch["label"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log(f"{split}_loss", loss)

        if split in ["val", "test"]:
            preds = y_hat.argmax(dim=1)
            acc = self.accuracy(preds, y)
            self.log(f"{split}_acc", acc, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)
