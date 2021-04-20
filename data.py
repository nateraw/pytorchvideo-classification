from pathlib import Path

import pytorch_lightning as pl
from pytorchvideo.data.clip_sampling import UniformClipSampler
from pytorchvideo.data.encoded_video_dataset import EncodedVideoDataset
from pytorchvideo.transforms import ApplyTransformToKey, Normalize, ShortSideScale, UniformTemporalSubsample
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo


class UCF11(pl.LightningDataModule):

    def __init__(
        self,
        root="./",
        batch_size=32,
        num_workers=8,
        num_holdout_scenes=1,
        side_size = 256,
        crop_size = 256,
        clip_mean = (0.45, 0.45, 0.45),
        clip_std = (0.225, 0.225, 0.225),
        num_frames = 8,
        sampling_rate = 8,
        frames_per_second = 30
    ):
        super().__init__()

        self.root = Path(root) / 'action_youtube_naudio'
        assert self.root.exists(), "Dataset not found."
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_holdout_scenes = num_holdout_scenes
        self.side_size = side_size
        self.mean = clip_mean
        self.std = clip_std
        self.crop_size = crop_size
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.frames_per_second = frames_per_second
        self.clip_duration = (self.num_frames * self.sampling_rate) / self.frames_per_second

        self.classes = [x.name for x in self.root.glob("*") if x.is_dir()]
        self.id_to_label = dict(zip(range(len(self.classes)), self.classes))
        self.class_to_label = dict(zip(self.classes, range(len(self.classes))))
        self.num_classes = len(self.classes)


        # TODO - too many repeated .glob calls here.
        self.train_paths = []
        self.val_paths = []
        self.holdout_scenes = {}
        for c in self.classes:

            # 'v_biking_01', 'v_biking_02', 'v_biking_03' 
            all_class_scenes = sorted(set(x.name for x in (self.root / c).glob("*") if x.is_dir() and x.name != 'Annotation'))

            # Keep last, (v_biking_03) as one we use for validation
            holdout_scenes = all_class_scenes[-num_holdout_scenes:]
            self.holdout_scenes[c] = holdout_scenes

            # [(action_youtube_naudio/biking/v_biking_01/v_biking_01_01.avi, <ID>) ... ]
            label_paths = [(v, {"label": self.class_to_label[c]}) for v in (self.root / c).glob("**/*.avi")]

            self.train_paths.extend([x for x in label_paths if x[0].parent.name not in holdout_scenes])
            self.val_paths.extend([x for x in label_paths if x[0].parent.name in holdout_scenes])


        # TODO - this is very specific to the pretrained model we chose
        self.transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(self.num_frames),
                    Lambda(lambda x: x / 255.0),
                    Normalize(self.mean, self.std),
                    ShortSideScale(size=self.side_size),
                    CenterCropVideo(crop_size=(self.crop_size, self.crop_size)),
                ]
            ),
        )

    def train_dataloader(self):
        ds = EncodedVideoDataset(
            self.train_paths,
            UniformClipSampler(self.clip_duration),
            decode_audio=False,
            transform=self.transform,
            video_sampler=RandomSampler,
        )
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        ds = EncodedVideoDataset(
            self.val_paths,
            UniformClipSampler(self.clip_duration),
            decode_audio=False,
            transform=self.transform,
            video_sampler=RandomSampler,
        )
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers)
