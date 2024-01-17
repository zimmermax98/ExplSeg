"""
Pytorch Lightning DataModule for training prototype segmentation model on Cityscapes and SUN datasets
"""
import multiprocessing
import os

import gin
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from segmentation.dataset import PatchClassificationDataset
from settings import data_path


# Try this out in case of high RAM usage:
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')


# noinspection PyAbstractClass
@gin.configurable(denylist=['batch_size'])
class PatchClassificationDataModule(LightningDataModule):
    def __init__(self, batch_size: int, dataloader_n_jobs: int = gin.REQUIRED,
                 train_key: str = 'train'):
        super().__init__()
        self.dataloader_n_jobs = dataloader_n_jobs if dataloader_n_jobs != -1 else multiprocessing.cpu_count()
        self.batch_size = batch_size
        self.train_key = train_key

    def prepare_data(self):
        if not os.path.exists(os.path.join(data_path, 'annotations')):
            raise ValueError("Please download dataset and preprocess it using 'preprocess.py' script")

    def get_data_loader(self, dataset: PatchClassificationDataset, **kwargs) -> DataLoader:
        if 'batch_size' in kwargs:
            return DataLoader(
                dataset=dataset,
                shuffle=not dataset.is_eval,
                num_workers=self.dataloader_n_jobs,
                **kwargs
            )
        return DataLoader(
            dataset=dataset,
            shuffle=not dataset.is_eval,
            num_workers=self.dataloader_n_jobs,
            batch_size=self.batch_size,
            **kwargs
        )

    def train_dataloader(self, **kwargs):
        train_split = PatchClassificationDataset(
            split_key=self.train_key,
            is_eval=False,
        )
        return self.get_data_loader(train_split, **kwargs)

    def val_dataloader(self, **kwargs):
        val_split = PatchClassificationDataset(
            split_key='val',
            is_eval=True,
        )
        return self.get_data_loader(val_split, **kwargs)

    def test_dataloader(self, **kwargs):
        test_split = PatchClassificationDataset(
            split_key='val',  # We do not have test set for cityscapes
            is_eval=True,
        )
        return self.get_data_loader(test_split, **kwargs)

    def train_push_dataloader(self, **kwargs):
        train_split = PatchClassificationDataset(
            split_key='train',
            is_eval=True,
            push_prototypes=True
        )
        return self.get_data_loader(train_split, **kwargs)
