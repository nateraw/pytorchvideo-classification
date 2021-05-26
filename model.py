import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy


class Classifier(pl.LightningModule):

    def __init__(self, model, lr: float = 2e-4, **kwargs):
        super().__init__()
        self.save_hyperparameters('lr', *list(kwargs))
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        outputs = self(batch["video"])
        loss = self.loss_fn(outputs, batch["label"])
        self.log(f"train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["video"])
        loss = self.loss_fn(outputs, batch["label"])
        self.log(f"val_loss", loss)
        acc = self.val_acc(outputs.argmax(1), batch["label"])
        self.log(f"val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
