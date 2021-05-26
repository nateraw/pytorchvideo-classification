# PyTorch Video Classification

<a href="https://colab.research.google.com/gist/nateraw/45c4ea2d00db2c6432d2854b759281f7/ptv-classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Usage

#### Finetuning Torchhub Models

```python
import torch
import pytorch_lightning as pl
from pytorchvideo.models import create_res_basic_head

from model import Classifier
from data import make_ucf11_datamodule

# Download data, prepare splits
dm = make_ucf11_datamodule()

# Load a model from Torchhub, freeze its backbone, and replace its classification head
model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
model.blocks[:-1].requires_grad_(False)
model.blocks[-1] = create_res_basic_head(in_features=2048, out_features=dm.num_labels)

# Train w/ PyTorch Lightning
classifier = Classifier(model, lr=2e-4)
trainer = pl.Trainer(gpus=1, precision=16, max_epochs=4)
trainer.fit(classifier, dm)
```

#### Use Your Own Models

```python
import torch
import pytorch_lightning as pl
from pytorchvideo.models import create_res_basic_head

from model import Classifier
from data import make_ucf11_datamodule

# Download data, prepare splits
dm = make_ucf11_datamodule()

# Any torch model that accepts video tensors + outputs class predictions
model = ...

# Train w/ PyTorch Lightning
classifier = Classifier(model, lr=2e-4)
trainer = pl.Trainer(gpus=1, precision=16, max_epochs=4)
trainer.fit(classifier, dm)
```
