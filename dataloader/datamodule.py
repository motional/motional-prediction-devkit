import logging
from typing import Callable, Optional

from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader

from dataloader.dataset import MotionalDataset

logger = logging.getLogger(__name__)

class MotionalDataModule(LightningDataModule):

    def __init__(self,
                 root: str = 'prediction',
                 map_root: str = 'map',
                 train_batch_size: int = 1,
                 val_batch_size: int = 1,
                 shuffle: bool = True,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 **kwargs) -> None:
        super(MotionalDataModule, self).__init__()
        self.root = root
        self.map_root = map_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers
        self.kwargs = kwargs

    def prepare_data(self) -> None:
        MotionalDataset(self.root, 'train', map_root=self.map_root, **self.kwargs)
        MotionalDataset(self.root, 'val', map_root=self.map_root, **self.kwargs)
        try:
            MotionalDataset(self.root, 'test', map_root=self.map_root, **self.kwargs)
        except:
            # it's fine to not have this when developing. used for test
            pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = MotionalDataset(self.root, 'train' , map_root=self.map_root, **self.kwargs)
        self.val_dataset = MotionalDataset(self.root, 'val', map_root=self.map_root, **self.kwargs)
        try:
            self.test_dataset = MotionalDataset(self.root, 'test', map_root=self.map_root, **self.kwargs)
        except:
            logger.info("skip building test set.")
            self.test_dataloader = None
            pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers, drop_last=self.kwargs["drop_last"])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, drop_last=False)

    def test_dataloader(self):
        if self.test_dataloader is None:
            return None
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, drop_last=False)
