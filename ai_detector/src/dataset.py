from typing import Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.robustness import apply_degradation_pil


class ImageBinaryDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform=None,
        return_path: bool = False,
        degradation: Optional[dict] = None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.return_path = return_path
        self.degradation = degradation

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        image = apply_degradation_pil(image, self.degradation)

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label"])
        path = row["path"]

        if self.return_path:
            return image, label, path
        return image, label


def build_image_transform(image_size: int = 224, is_train: bool = False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
