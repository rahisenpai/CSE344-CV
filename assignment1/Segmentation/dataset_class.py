import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CAMVidDataset(Dataset):
    """
    Implements a custom dataset (similar to the one in PyTorch) to handle data preparation of
    CAMVid Dataset for training Neural Networks for Image Segmentation Task. This dataset is
    compatible with PyTorch's DataLoader.
    """

    @classmethod
    def initialize_class_vars(cls, root: str) -> None:
        """
        Initialize class variables before creating any instance of this Dataset object.
        """
        class_dict = pd.read_csv(os.path.join(root, "class_dict.csv"))
        cls.INDEX2LABEL = {idx: row[0] for idx, row in enumerate(class_dict.values)}
        cls.LABEL2INDEX = {lbl: idx for idx, lbl in cls.INDEX2LABEL.items()}
        cls.INDEX2COLOR = {idx: (row[1], row[2], row[3]) for idx, row in enumerate(class_dict.values)}
        cls.COLOR2INDEX = {clr: idx for idx, clr in cls.INDEX2COLOR.items()}

    def __init__(self, root: str, train: bool, transform: transforms.Compose =None,
                 mask_transform: transforms.Compose =None):
        """
        Initializes the Wildlife dataset.
        """
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_dir, self.mask_dir = None, None
        self.image_dir = os.path.join(root, "train" if train else "test_images")
        self.mask_dir = os.path.join(root, "train_labels" if train else "test_labels")
        self.images = sorted(os.listdir(os.path.join(self.image_dir)))
        self.masks = sorted(os.listdir(os.path.join(self.mask_dir)))

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.images)
    
    def mask_to_class(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Converts RGB mask to class index mask.
        """
        mask = np.array(mask)
        label_mask = np.zeros(mask.shape[:2], dtype=np.int64)
        for color, idx in self.COLOR2INDEX.items():
            label_mask[np.all(mask == np.array(color), axis=-1)] = idx
        return torch.from_numpy(label_mask).long()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        """
        Returns the image and mask at the given index.
        """
        image = Image.open(os.path.join(self.image_dir, self.images[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.masks[idx])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = self.mask_to_class(mask)
        return image, mask


""" Refernces
- https://github.com/rahisenpai/CSE344-CV/blob/main/assignment1/Classification/dataset_class.py
"""