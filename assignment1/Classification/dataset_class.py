import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class WildlifeDataset(Dataset):
    """
    Implements a custom dataset (similar to the one in PyTorch) to handle data preparation of
    Russian Wildlife Dataset for training Neural Networks for Image Classification Task. This
    dataset is compatible with PyTorch's DataLoader.
    """
    LABEL2INDEX: dict[str, int] = {
        'amur_leopard': 0,
        'amur_tiger': 1,
        'birds': 2,
        'black_bear': 3,
        'brown_bear': 4,
        'dog': 5,
        'roe_deer': 6,
        'sika_deer': 7,
        'wild_boar': 8,
        'people': 9
    } #a class dictionary to map string labels to numbers or indices

    #a class dictionary to map number labels to strings
    INDEX2LABEL: dict[int, str] = {index: label for label, index in LABEL2INDEX.items()}

    def __init__(self, root: str, transform: transforms.Compose =None, label_transform: transforms.Compose =None):
        """
        Initializes the Wildlife dataset.
        """
        self.root = root
        self.transform = transform
        self.label_transform = label_transform
        self.images = []
        self.labels = []
        #add image paths and labels in lists
        for label in WildlifeDataset.LABEL2INDEX:
            label_dir = os.path.join(self.root, label)
            for image in os.listdir(label_dir):
                self.images.append(os.path.join(label_dir, image)) #add the relative path to the image
                self.labels.append(WildlifeDataset.LABEL2INDEX[label]) #add the number mapped to label

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        """
        Returns the image and label at the given index.
        """
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image)
        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        return image, label


""" Refernces
- https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
- https://www.tutorialspoint.com/python/os_file_methods.htm
- https://github.com/rahisenpai/CSE343-ML/blob/main/asgn4/code_c.ipynb
- https://github.com/rahisenpai/CSE556-NLP/blob/main/assignment1/task3.py
"""