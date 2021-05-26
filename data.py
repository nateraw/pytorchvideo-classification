import itertools
import requests
from pathlib import Path
from random import shuffle
from shutil import unpack_archive

import pytorch_lightning as pl
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)


class LimitDataset(Dataset):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos


class LabeledVideoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_paths,
        val_paths,
        clip_duration: int = 2,
        batch_size: int = 4,
        num_workers: int = 2,
        **kwargs
    ):
        super().__init__()
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.clip_duration = clip_duration
        self.num_labels = len({path[1] for path in train_paths._paths_and_labels})
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.train_transforms = ApplyTransformToKey(
            key='video',
            transform=Compose(
                [
                    UniformTemporalSubsample(8),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(224),
                    RandomHorizontalFlip(p=0.5),
                ]
            )
        )
        self.val_transforms = ApplyTransformToKey(
            key='video',
            transform=Compose(
                [
                    UniformTemporalSubsample(8),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    ShortSideScale(256),
                    CenterCrop(224)
                ]
            )
        )

    def train_dataloader(self):
        self.train_dataset = LimitDataset(
            LabeledVideoDataset(
                self.train_paths,
                clip_sampler=make_clip_sampler('random', self.clip_duration),
                decode_audio=False,
                transform=self.train_transforms,
            )
        )
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        self.val_dataset = LimitDataset(
            LabeledVideoDataset(
                self.val_paths,
                clip_sampler=make_clip_sampler('uniform', self.clip_duration),
                decode_audio=False,
                transform=self.val_transforms,
            )
        )
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


_mini_kinetics_url = 'https://pl-flash-data.s3.amazonaws.com/kinetics.zip'
def make_mini_kinetics_datamodule(root='./', **kwargs):
    kinetics_path = Path(root) / 'kinetics'
    if not kinetics_path.exists():
        download_and_unzip(_mini_kinetics_url, root)
        
    return LabeledVideoDataModule(
        LabeledVideoPaths.from_path(kinetics_path / 'train'),
        LabeledVideoPaths.from_path(kinetics_path / 'val'),
        **kwargs
    )


_ucf11_url = 'https://www.crcv.ucf.edu/data/YouTube_DataSet_Annotated.zip'
def make_ucf11_datamodule(root='./', **kwargs):
    data_path = Path(root) / 'action_youtube_naudio'
    if not data_path.exists():
        download_and_unzip(_ucf11_url, root, False)

    # Collect all class names, scene folders, and label2id mapping
    classes = sorted(x.name for x in data_path.glob("*") if x.is_dir())
    label2id = {}
    scene_folders = []
    for class_id, class_name in enumerate(classes):
        label2id[class_name] = class_id
        class_folder = data_path / class_name
        scene_folders.extend(list(filter(Path.is_dir, class_folder.glob('v_*'))))

    shuffle(scene_folders)

    num_train_scenes = int(0.8 * len(scene_folders))
    train_paths, val_paths = [], []
    for i, scene in enumerate(scene_folders):
        class_id = label2id[scene.parent.name]
        labeled_paths = [(video, class_id) for video in scene.glob('*.avi')]
        if i < num_train_scenes:
            train_paths.extend(labeled_paths)
        else:
            val_paths.extend(labeled_paths)

    return LabeledVideoDataModule(
        LabeledVideoPaths(train_paths),
        LabeledVideoPaths(val_paths),
        label2id=label2id,
        classes=classes,
        **kwargs
    )


def download_and_unzip(url, data_dir="./", verify=True):
    """Download a zip file from a given URL and unpack it within data_dir.

    Args:
        url (str): A URL to a zip file.
        data_dir (str, optional): Directory where the zip will be unpacked. Defaults to "./".
        verify (bool, optional): Whether to verify SSL certificate when requesting the zip file. Defaults to True.
    """
    data_dir = Path(data_dir)
    zipfile_name = url.split("/")[-1]
    data_zip_path = data_dir / zipfile_name
    data_dir.mkdir(exist_ok=True, parents=True)

    if not data_zip_path.exists():
        resp = requests.get(url, verify=verify)

        with data_zip_path.open("wb") as f:
            f.write(resp.content)

    unpack_archive(data_zip_path, extract_dir=data_dir)
